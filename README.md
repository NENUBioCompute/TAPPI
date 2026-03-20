# TAPPI

## Installation

First clone the repository from GitHub:

```bash
git clone git@github.com:NENUBioCompute/TAPPI.git
cd TAPPI
```

Then install the required dependencies:

```bash
pip install -r requirements.txt
```

It is recommended to use a Python environment with **Python ≥ 3.8** and **PyTorch ≥ 2.0**.

Download the model parameters from the [Figshare](https://figshare.com/ndownloader/files/62900644) and place them in the root directory of the project:

After installing the dependencies, the environment will be ready for running the TAPPI model and related experiments.

## Inference Example

Run the following command to make a prediction:

```bash
python predict.py \
    --seq VSFRYIFGLPPLILVLLPVASSDCDIEGKDGKQYE \
    --mutation Y5R \
    --partner PPLILVLLPVASSDCDIEGKDGK
```

### Evaluation Instructions

The evaluation results presented in this manuscript can be reproduced using the provided dataset and code. 

1. **Download the test dataset**  
   Download the test dataset from [Figshare](https://figshare.com/ndownloader/files/62913865) and place it in the root directory of this repository.

2. **Run the evaluation**  
Once the dataset is in place, run the evaluation pipeline with:

```bash
python -m train.test
```

The obtained results are consistent with the conclusions derived from training using five-fold cross-validation with a fixed random seed for dataset splitting into training, validation, and test sets. The five-fold cross-validation dataset, as well as the data splitting and training procedures, can be reproduced by sequentially executing the [provided scripts](https://github.com/NENUBioCompute/TAPPI/blob/main/train/train_five_cross.ipynb).

### Model Training
The model can be trained by executing:

```bash
python -m train.train
```

## introduction
 a deep learning framework utilizes PLM to predict protein-protein interaction variation trends 
 <div align="center">
    <img src="docs/base_back.png", width="800">
</div>

**TAPPI** is a model for predicting the effects of mutations on protein–protein interactions (PPI). It classifies PPI variations into four categories: *Increasing*, *Decreasing*, *Disrupting*, and *No_Effect*.

## extensions
We introduce extensions to improve class balance and to explicitly handle multi-point mutations across arbitrary distances.
