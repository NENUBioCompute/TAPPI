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

## introduction
 a deep learning framework utilizes PLM to predict protein-protein interaction variation trends 
 <div align="center">
    <img src="docs/base_back.png", width="800">
</div>

* train&benchmark -> **source code** of training and benchmark evaluation
* external -> Apply TAPPI to **external datasets** and ***pathogenic analyse**
* ## Pre-trained Models and Test Dataset
The pre-trained models and test datasets can be loaded from [Figshare](https://figshare.com/articles/dataset/x_test_fold_mirror_multi_csv/29637008).
## Model Evaluation
You can easily understand the model's performance by executing the notebook available at [mippi_simply_variation.ipynb](https://github.com/NENUBioCompute/TAPPI/blob/main/mippi_simply_variation.ipynb).
# Requirement
* python 3.8+
* pytorch 1.12.0+
* numpy
* pandas
* fair-esm
