# Helix encoder: A compound-protein interaction prediction model specifically designed for Class A GPCRs

![fotAbstract (1)](https://user-images.githubusercontent.com/67744833/226157282-646a1e6e-77b9-462c-b2c1-2bcae33ed700.png)

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
## Using
- `mol_featurizer_for_helix.py`: generate input for Helix encoder
- `helix_encoder_main.py`: trains Helix encoder model

## Author

## Citation
