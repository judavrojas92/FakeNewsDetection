# Fake News Detection Using ISOT Dataset

This project is a replication and extension of the baseline proposed by Hoy & Koulouri (2022) on fake news detection. It was developed as part of the MSc&T "Data and Economics for Public Policy" at École Polytechnique and ENSAE, and it systematically explores feature engineering, classification, and generalization under distributional shift using the ISOT Fake News Dataset.

## Objective

The project aims to replicate and stress-test the fake news detection pipelines evaluated by Hoy & Koulouri (2022). Beyond in-domain validation, a novel **source-bias removal experiment** is introduced to simulate real-world generalization challenges by decoupling source identity from label information.

---

## Dataset

- **ISOT Fake News Dataset**  
  - **Real news** from Reuters.com  
  - **Fake news** from unreliable news sources  
  - Preprocessed to remove stopwords, punctuation, and perform lemmatization  
  - Only the `text` field was used (no titles, dates, or subjects)

---

## Algorithms

### Feature Pipelines

Five text representation pipelines were explored:

- Bag of Words (`CountVectorizer`)
- TF-IDF (`TfidfVectorizer`)
- Word2Vec (`spaCy` vectors)
- DistilBERT (Transformers-based)
- Linguistic Cues (e.g. punctuation density, pronoun ratio, etc.)

### Classifiers

Each representation was evaluated using six classifiers:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- AdaBoost
- Neural Network (MLP)

All evaluations used **stratified 2-fold cross-validation**.

---

## Experiments

### 1. Paper Replication
Each feature-classifier combination was tested under standard conditions to reproduce Hoy & Koulouri’s results. Ensemble models using TF-IDF and BoW achieved **F1 > 0.99**.

### 2. Source-Bias Removal
To test model generalization, a new setup was created where portions of the *Reuters* data were excluded from training. This revealed performance drops across all pipelines, confirming the existence of **source-label shortcuts** in the ISOT dataset.

---

## Visualizations

- Document length distributions
- Word clouds (real vs. fake)
- Most frequent words per class
- Bar plots comparing F1, Accuracy, Precision, and Recall under random vs. source-biased splits

Plots are available in the `figures/` directory.

---

## Structure

- `notebooks/` – All experiment notebooks
- `data/` – Preprocessed ISOT data
- `figures/` – All visual outputs (plots, word clouds)
- `report.pdf` – Final report submitted for evaluation
- `README.md` – This file

---

## Conclusion

- Shallow pipelines (TF-IDF, BoW) with ensemble models are strong in-domain.
- Deep models (BERT, Word2Vec) generalize better under source shifts.
- Linguistic Cues are less reliable when used in isolation.
- Generalization requires **source-debiased** evaluation setups.

---

## Contact

Julian David Rojas Rojas  
julian.rojas@ensae.fr  MSc&T Data and Economics for Public Policy – École Polytechnique / ENSAE

---

## Acknowledgments

This project was completed as part of the MSc&T DEPP program. I thank the course coordinators for their guidance and the original authors Hoy & Koulouri (2022) for making their methodology reproducible.


## **File Structure**

```
├── README.md  
├── LICENSE <- Repository license (default: MIT)  
├── Code  
│   └── Main code for the Report: Fake News Detection Using ISOT Dataset   
├── Report 
│   └── Final report: Fake News Detection Using ISOT Dataset 

```
