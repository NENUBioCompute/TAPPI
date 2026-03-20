# TAPPI (Weighted GMH Loss Version)

This repository provides an improved version of TAPPI trained with a **weighted GMH loss**, aiming to achieve **more balanced prediction performance**.

---

## 🔧 Installation

Please follow the official installation instructions from the original repository:

👉 https://github.com/NENUBioCompute/TAPPI

---

## ⬇️ Download Parameters

Before running prediction, download the model parameters:

```bash
wget https://figshare.com/ndownloader/files/62952412
```

---

## 🚀 Example Prediction

Run a sample prediction with the following command:

```bash
python predict.py \
    --seq VSFRYIFGLPPLILVLLPVASSDCDIEGKDGKQYE \
    --mutation Y5R I6A G8Y \
    --partner PPLILVLLPVASSDCDIEGKDGK
```
