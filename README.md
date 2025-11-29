# Forecasting-Mental-Health-Distress-in-Online-Forums-Using-Linguistic-and-Temporal-Models

This repository contains all code, data-processing scripts, and analysis notebooks for my graduate research project at the University of Adelaide.
The work explores how linguistic signals, temporal emotional trajectories, and LLM-derived features can be combined to predict early signs of mental health distress in online forum users Beyond Blue.

The pipeline integrates:

* LLM-based emotion and symptom extraction (RoBERTa / Empath / custom V-A weights)

* Circumplex emotion mapping

* Weekly forecasting with ARX

* SHAP interpretability

* Uncertainty quantification

* Extreme-change detection

* Case studies for stable vs volatile users

## Technologies
* Python 3.10.14

## Installation
* [Mental-RoBERTa](https://huggingface.co/mental/mental-roberta-base)
* [NRC-VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html) or other lexicon
