# Fake News Detection Using ISOT Dataset

This project is a replication and extension of the baseline proposed by Hoy & Koulouri (2022) on fake news detection. It was developed as part of the MSc&T "Data and Economics for Public Policy" at École Polytechnique and ENSAE, and it systematically explores feature engineering, classification, and generalization under distributional shift using the ISOT Fake News Dataset.

## Objective

The project aims to replicate and stress-test the fake news detection pipelines evaluated by Hoy & Koulouri (2022). Beyond in-domain validation, a novel **source-bias removal experiment** is introduced to simulate real-world generalization challenges by decoupling source identity from label information.


# Project Overview: Fake News Detection using ISOT Dataset

This project replicates and expands upon the fake news detection baseline introduced by Hoy and Koulouri (2022), using the ISOT dataset as the sole data source. The main goal is to assess the replicability and robustness of classic and modern natural language processing (NLP) techniques in detecting fake news articles. Five different feature extraction techniques—Bag of Words (BoW), TF-IDF, Word2Vec (spaCy), DistilBERT, and Linguistic Cues—are combined with six classifiers (Logistic Regression, SVM, Random Forest, Gradient Boosting, AdaBoost, Neural Network), evaluated under 2-fold stratified cross-validation.

The paper also introduces a source-bias removal experiment to simulate domain shift by withholding all Reuters news (true news) from the training set. This second part analyzes model generalization beyond standard random splits and reveals significant performance drops, reinforcing the importance of distribution-aware evaluation in fake news detection research.


# Code Overview: `FakeNewsDetection.ipynb`

The notebook starts by downloading the ISOT dataset via KaggleHub, followed by extensive data preprocessing steps:
- Lowercasing, punctuation removal, lemmatization (via NLTK)
- Minimal cleaning for word embeddings and BERT-style pipelines

The code is modular and organized into the following major blocks:

### Feature Extraction Pipelines
Implements:
- TF-IDF and CountVectorizer (BoW)
- spaCy word vectors (averaged)
- DistilBERT (Hugging Face Transformers)
- Handcrafted linguistic cues (e.g., avg sentence length, punctuation usage)

### Model Evaluation
Each feature type is tested with 6 classifiers using scikit-learn:
- Logistic Regression, SVM, Neural Network (MLP), Random Forest, Gradient Boosting, AdaBoost
- Evaluation metrics include Accuracy, Precision, Recall, and F1-score under 2-fold CV

### Generalization Testing
A custom experiment simulates a domain shift:
- Training on fake + partial real news, testing on withheld Reuters real articles
- Highlights performance degradation and model robustness to distribution shifts

###  Visualization and Reporting
Generates:
- Barplots (F1, Acc, etc.)
- Summary tables per feature+classifier combination
- Interpretations of generalization gaps

## For questions or contributions:  

Julian David Rojas Rojas  
julian.rojas@ensae.fr  
MSc&T Data and Economics for Public Policy – École Polytechnique / ENSAE

## Acknowledgments

This project was completed as part of the MSc&T DEPP program. I thank our NLP professor Christopher Kermorvant and the original authors Hoy & Koulouri (2022) for making their methodology reproducible.


## **File Structure**

```
├── README.md  
├── LICENSE <- Repository license (default: MIT)  
├── Code  
│   └── Main code for the Report: Fake News Detection Using ISOT Dataset   
├── Report 
│   └── Final report: Fake News Detection Using ISOT Dataset 

```
