# Helix encoder: A compound-protein interaction prediction model specifically designed for Class A GPCRs


<div align="center">
    <img src="https://user-images.githubusercontent.com/67744833/226157282-646a1e6e-77b9-462c-b2c1-2bcae33ed700.png" alt="figure">
</div>


## Dependencies
- Python = 3.7.10
- pytorch >= 1.2.0
- numpy = 1.20.2
- RDkit = 2020.09.1
- pandas = 1.3.4
- Gensim >=3.4.0

## Setup
- Clone [TransformerCPI](https://github.com/lifanchen-simm/transformerCPI)
- Place each file in this repository in the TransformerCPI directory

## Data
- `/csvData`
  - csv files of protein sequences, compound SMILES, and interaction data used in the experiments
- `/data`
  - Text data as input for mol_featurizer
  - data format
    - A text file containing compound SMILES, protein sequences, and interactions (0 or 1) in this order, separated by spaces. Protein sequences of each transmembrane region and extracellular loop region are also separated by spaces.
```
O=C(OCn1ncc(Br)c(Br)c1=O)c1c(F)cccc1F GLSVAASCLVVLENLLVLAAI LVNITLSDLLTGAAYLANVLL WFLREGLLFTALAASTFSLLF VYGFIGLCWLLAALLGMLPLL FCLVIFAGVLATIMGLYGAIF VLMILLAFLVCWGPLFGLLLA MDWILALAVLNSAVNPIIYSF 1
```
- `/dataset`
  - Directory where data embedded by mol_featurizer is stored
  
## How to use
### Embedding
- Generate input for Helix encoder. 
  -  `python mol_featurizer_for_TM.py`  

### Model training
- Trains Helix encoder model.
  - `pytohn helix_encoder_main.py`    

### Predict
- A trained model, Helix encoder (TM + ECL2), exists in this repository (`/output/model/helixEncoder_TM_ECL2`). If you want to use this model to predict your own data, use the following.

1. Place the data you want to predict in /data/.
2. At mol_featurizer_for_TM, place the embedding vector in /dataset/.
3. For prediction, run `python predict.py`

## Reference
1. Lifan Chen, Xiaoqin Tan, Dingyan Wang, Feisheng Zhong, Xiaohong Liu, Tianbiao Yang, Xiaomin Luo, Kaixian Chen, Hualiang Jiang, Mingyue Zheng, TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments, Bioinformatics, Volume 36, Issue 16, 15 August 2020, Pages 4406–4414, https://doi.org/10.1093/bioinformatics/btaa524


