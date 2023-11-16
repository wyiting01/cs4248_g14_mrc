# Welcome to `model/`

As the file size of model weights are too large, we have uploaded them to Google Shared Drive. 
If interested, feel free to access the weights for our baseline models [here](https://drive.google.com/drive/folders/1EEMnIqmGX_sSOeRpoAJjFOv8HW81W7rq?usp=sharing).

Our ensemble models are built based on these weights too!

## Available Weights:
1. `bilstm.pt` : biLSTM weights trained with 100% dataset
2. `roberta_optuna.pt` : Roberta weights trained with 80% of the dataset
3. `roberta.pt` : Roberta weights trained with 100% of the dataset
4. `xlnet_optuna.pt` : XLNet weights trained with 80% of the dataset
5. `xlnet.pt` : XLNet weights trained with 100% of the dataset

> Some models are trained only with 80% of the dataset for our ensemble technique, explained [here](https://github.com/wyiting01/cs4248_g14_mrc#unequal-weightage) under **"Section 3. Weighting based on Optuna"**
