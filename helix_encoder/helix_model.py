import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from Radam import *
from lookahead import Lookahead

torch.set_default_tensor_type('torch.cuda.FloatTensor')


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads
                                                   ])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads,
                   self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads,
                   self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads,
                   self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim,
                 decoder_layer, self_attention, positionwise_feedforward,
                 dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList([
            decoder_layer(hid_dim, n_heads, pf_dim, self_attention,
                          positionwise_feedforward, dropout, device)
            for _ in range(n_layers)
        ])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam([{
            'params': weight_p,
            'weight_decay': weight_decay
        }, {
            'params': bias_p,
            'weight_decay': 0
        }],
                                     lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms, proteins, labels = [], [], [], []
        for data in dataset:
            i = i + 1
            atom, adj, protein, label = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels.append(label)
            if i % 8 == 0 or i == N:
                data_pack = pack(atoms, adjs, proteins, labels, device)
                loss = self.model(data_pack)
                # loss = loss / self.batch
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                adjs, atoms, proteins, labels = [], [], [], []
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels = [], [], [], []
                atom, adj, protein, label = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                data = pack(atoms, adjs, proteins, labels, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(
                    data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        return AUC, PRC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


####################################################################
class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, input, adj):
        # input =[batch,num_node, atom_dim]
        # adj = [batch,num_node, num_node]
        support = torch.matmul(input, self.weight)
        # support =[batch,num_node,atom_dim]
        output = torch.bmm(adj, support)
        # output = [batch,num_node,atom_dim]
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len,
                   protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    def forward(self, compound, adj, protein, atom_num, protein_num):
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 100]
        compound_max_len = compound.shape[1]
        protein_max_len = protein[0].shape[
            1]  ###############################################################
        #protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num,
                                                      compound_max_len,
                                                      protein_max_len)
        compound = self.gcn(compound, adj)
        # compound = torch.unsqueeze(compound, dim=0)
        # compound = [batch size=1 ,atom_num, atom_dim]

        # protein = torch.unsqueeze(protein, dim=0)
        # protein =[ batch size=1,protein len, protein_dim]
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]

        out = self.decoder(compound, enc_src, compound_mask, protein_mask)
        # out = [batch size, 2]
        # out = torch.squeeze(out, dim=0)
        return out

    def __call__(self, data, train=True):

        compound, adj, protein, correct_interaction, atom_num, protein_num = data
        # compound = compound.to(self.device)
        # adj = adj.to(self.device)
        # protein = protein.to(self.device)
        # correct_interaction = correct_interaction.to(self.device)
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction = self.forward(compound, adj, protein,
                                                 atom_num, protein_num)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            #compound = compound.unsqueeze(0)
            #adj = adj.unsqueeze(0)
            #protein = protein.unsqueeze(0)
            #correct_interaction = correct_interaction.unsqueeze(0)
            predicted_interaction = self.forward(compound, adj, protein,
                                                 atom_num, protein_num)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


def pack(atoms, adjs, proteins, labels, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    protein_num = []
    for p in proteins:
        for protein in p:
            protein_num.append(protein.shape[0])
            if protein.shape[0] >= proteins_len:
                proteins_len = protein.shape[0]
    #print(proteins_len)
    atoms_new = torch.zeros((N, atoms_len, 34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = atom
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len, device=device)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    #############################
    proteins_news_7 = [
        torch.zeros((N, proteins_len, 100), device=device) for i in range(7)
    ]
    i = 0
    for p in proteins:
        for index, protein in enumerate(p):
            a_len = protein.shape[0]
            proteins_news_7[index][i, :a_len, :] = protein
        i += 1
    ###############################
    labels_new = torch.zeros(N, dtype=torch.long, device=device)

    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_news_7, labels_new, atom_num,
            protein_num)


# class Encoder(nn.Module):
#     """protein feature extraction."""
#     def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout,
#                  device):
#         super().__init__()

#         assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

#         self.input_dim = protein_dim
#         self.hid_dim = hid_dim
#         self.kernel_size = kernel_size
#         self.dropout = dropout
#         self.n_layers = n_layers
#         self.device = device
#         #self.pos_embedding = nn.Embedding(1000, hid_dim)
#         self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
#         self.convs = nn.ModuleList([
#             nn.Conv1d(hid_dim,
#                       2 * hid_dim,
#                       kernel_size,
#                       padding=(kernel_size - 1) // 2)
#             for _ in range(self.n_layers)
#         ])  # convolutional layers
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(self.input_dim, self.hid_dim)
#         self.gn = nn.GroupNorm(8, hid_dim * 2)
#         self.ln = nn.LayerNorm(hid_dim)

#     def forward(self, protein):
#         #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
#         #protein = protein + self.pos_embedding(pos)
#         #protein = [batch size, protein len,protein_dim]
#         conv_input = self.fc(protein)
#         # conv_input=[batch size,protein len,hid dim]
#         #permute for convolutional layer
#         conv_input = conv_input.permute(0, 2, 1)
#         #conv_input = [batch size, hid dim, protein len]
#         for i, conv in enumerate(self.convs):
#             #pass through convolutional layer
#             conved = conv(self.dropout(conv_input))
#             #conved = [batch size, 2*hid dim, protein len]

#             #pass through GLU activation function
#             conved = F.glu(conved, dim=1)
#             #conved = [batch size, hid dim, protein len]

#             #apply residual connection / high way
#             conved = (conved + conv_input) * self.scale
#             #conved = [batch size, hid dim, protein len]

#             #set conv_input to conved for next loop iteration
#             conv_input = conved

#         conved = conved.permute(0, 2, 1)
#         # conved = [batch size,protein len,hid dim]
#         conved = self.ln(conved)
#         return conved


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size, dropout,
                 device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.multi_convs1 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs2 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs3 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs4 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs5 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs6 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])
        self.multi_convs7 = nn.ModuleList([
            nn.Conv1d(hid_dim,
                      2 * hid_dim,
                      kernel_size,
                      padding=(kernel_size - 1) // 2)
            for _ in range(self.n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc2 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc3 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc4 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc5 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc6 = nn.Linear(self.input_dim, self.hid_dim)
        self.fc7 = nn.Linear(self.input_dim, self.hid_dim)

        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)
        self.fuuly = nn.Linear(hid_dim * 7, hid_dim)
        self.position_fuuly = nn.Linear(7, 1)
        self.r = nn.ReLU()

    def forward(self, proteins):
        conv_inputs = []

        conv_input1 = self.fc1(proteins[0])
        conv_input1 = conv_input1.permute(0, 2, 1)
        conv_input2 = self.fc2(proteins[1])
        conv_input2 = conv_input2.permute(0, 2, 1)
        conv_input3 = self.fc3(proteins[2])
        conv_input3 = conv_input3.permute(0, 2, 1)
        conv_input4 = self.fc4(proteins[3])
        conv_input4 = conv_input4.permute(0, 2, 1)
        conv_input5 = self.fc5(proteins[4])
        conv_input5 = conv_input5.permute(0, 2, 1)
        conv_input6 = self.fc6(proteins[5])
        conv_input6 = conv_input6.permute(0, 2, 1)
        conv_input7 = self.fc7(proteins[6])
        conv_input7 = conv_input7.permute(0, 2, 1)

        conved_result = []
        #1
        for i, conv in enumerate(self.multi_convs1):
            conved1 = conv(self.dropout(conv_input1))
            conved1 = F.glu(conved1, dim=1)
            conved1 = (conved1 + conv_input1) * self.scale
            conv_input1 = conved1
        conved1 = conved1.permute(0, 2, 1)
        conved1 = self.ln(conved1)
        conved_result.append(conved1)

        #2
        for i, conv in enumerate(self.multi_convs2):
            conved2 = conv(self.dropout(conv_input2))
            conved2 = F.glu(conved2, dim=1)
            conved2 = (conved2 + conv_input2) * self.scale
            conv_input2 = conved2
        conved2 = conved2.permute(0, 2, 1)
        conved2 = self.ln(conved2)
        conved_result.append(conved2)

        #3
        for i, conv in enumerate(self.multi_convs3):
            conved3 = conv(self.dropout(conv_input3))
            conved3 = F.glu(conved3, dim=1)
            conved3 = (conved3 + conv_input3) * self.scale
            conv_input3 = conved3
        conved3 = conved3.permute(0, 2, 1)
        conved3 = self.ln(conved3)
        conved_result.append(conved3)

        #4
        for i, conv in enumerate(self.multi_convs4):
            conved4 = conv(self.dropout(conv_input4))
            conved4 = F.glu(conved4, dim=1)
            conved4 = (conved4 + conv_input4) * self.scale
            conv_input4 = conved4
        conved4 = conved4.permute(0, 2, 1)
        conved4 = self.ln(conved4)
        conved_result.append(conved4)

        #5
        for i, conv in enumerate(self.multi_convs5):
            conved5 = conv(self.dropout(conv_input5))
            conved5 = F.glu(conved5, dim=1)
            conved5 = (conved5 + conv_input5) * self.scale
            conv_input5 = conved5
        conved5 = conved5.permute(0, 2, 1)
        conved5 = self.ln(conved5)
        conved_result.append(conved5)

        #6
        for i, conv in enumerate(self.multi_convs6):
            conved6 = conv(self.dropout(conv_input6))
            conved6 = F.glu(conved6, dim=1)
            conved6 = (conved6 + conv_input6) * self.scale
            conv_input6 = conved6
        conved6 = conved6.permute(0, 2, 1)
        conved6 = self.ln(conved6)
        conved_result.append(conved6)

        #7
        for i, conv in enumerate(self.multi_convs7):
            conved7 = conv(self.dropout(conv_input7))
            conved7 = F.glu(conved7, dim=1)
            conved7 = (conved7 + conv_input7) * self.scale
            conv_input7 = conved7
        conved7 = conved7.permute(0, 2, 1)
        conved7 = self.ln(conved7)
        conved_result.append(conved7)
        
        # 単純な線形層での圧縮
        concated = torch.cat(conved_result, dim=2)
        encoded = self.fuuly(concated)
        encoded = self.r(encoded)
        encoded = self.dropout(encoded)

        # positionを考慮した圧縮
        # conved_result_4dim = [torch.unsqueeze(p, 3) for p in conved_result]
        # concated = torch.cat(conved_result_4dim, dim=3)
        # position_vec = self.position_fuuly(concated)
        # encoded = torch.squeeze(position_vec, 3)
        # encoded = self.r(encoded)
        # encoded = self.dropout(encoded)


        return encoded