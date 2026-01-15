# FIHGT
FI-HGT: High-Order Feature Interactions via Heterogeneous Graph Transformer for Deep Social Bot Detection
## Enviromment
```
python 3.7
scikit-learn 1.0.2
torch 1.8.1+cu111
torch_cluster-1.5.9
torch_scatter-2.0.6
torch_sparse-0.6.9
torch_spline_conv-1.2.1
torch-geometric 2.0.4
pytorch-lightning 1.5.0
```

## Train Model

To start training process:

Train FIHGT models
```shell script
python FIHGT.py 
```

Train GNN models
```shell script
python MGTAB-GNN.py 
```

##  Datasets download
For datasets, please visit the [Bot Datasets]([https://drive.google.com/drive/folders/1DXjb48SYKlv0KxZb_rq3OlaC6JAU-4P7?usp=sharing]).
After downloading these datasets, please unzip it into path "./Dataset".
