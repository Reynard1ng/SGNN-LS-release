## Large-Scale Spectral Graph Neural Networks via Laplacian Sparsification

This is the demo version of the implementation of Large-Scale Spectral Graph Neural Networks via Laplacian Sparsification.


## Environment Settings    
- pytorch 1.12.1
- torch-geometric 2.1.0
- ogb 1.3.5

## Reproduction
For simplicity, we provide the dataset Cora and the splits in ./data and ./data_saved.
You may run the following command within the correct environment to reproduce the result of GPR-LS and APPNP-LS on Cora presented in Table 1.

python training.py --dataset Cora --net LSGPR   --device 0 --lr 0.05 --prop_lr 0.05 --weight_decay 0.0005 --prop_wd 0.0005 --dropout 0.8 --alpha 0.9 --hidden 64 --K 10 --ec 3
python training.py --dataset Cora --net LSAPPNP --device 0 --lr 0.05 --prop_lr 0.05 --weight_decay 0.0005 --prop_wd 0.0005 --dropout 0.5 --alpha 0.1 --hidden 64 --K 5 --ec 10

For other datasets, you may download the raw data through PyG, and then execute the proposed preprocessor.py to generate the data splits.
For papers100M, you may execute the proposed preprocessor_large.py to convert the dataset into SparseTensor format.
These processes are just for convenience, which can be integrated into our main program without any extra computational burden.



