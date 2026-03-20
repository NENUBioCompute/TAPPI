# TAPPI-multi

This repository provides **TAPPI-multi**, an extension of TAPPI designed for multi-point mutation analysis. It is trained by extracting local features for each mutation, enabling the modeling of cooperative effects of multiple mutations at arbitrary distances.

---

## 🔧 Installation

Please follow the official installation instructions from the original repository:

👉 https://github.com/NENUBioCompute/TAPPI

---

## 🚀 Example Prediction

To predict the cooperative effects of multiple mutations at arbitrary distances, run the following command:

```bash
python predict.py \
    --seq VSFRYIFGLPPLILVLLPVASSDCDIEGKDGKQYE \
    --mutation Y5R I6A G8Y \
    --partner PPLILVLLPVASSDCDIEGKDGK
