# TAPPI-multi (Weighted GMH Loss Version)

This repository provides **TAPPI-multi**, an improved version of TAPPI trained with a **weighted GMH loss**, designed to achieve **more balanced prediction performance** and to model the cooperative effects of multi-point mutations.

---

## 🚀 Example Prediction

To predict the cooperative effects of multiple mutations at arbitrary distances, run the following command:

```bash
python predict.py \
    --seq VSFRYIFGLPPLILVLLPVASSDCDIEGKDGKQYE \
    --mutation Y5R I6A G8Y \
    --partner PPLILVLLPVASSDCDIEGKDGK
