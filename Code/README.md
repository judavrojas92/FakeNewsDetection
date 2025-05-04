# Notebook: Fake News Detection using ISOT Dataset  

This notebook performs an end-to-end analysis of fake news detection using the ISOT dataset. It replicates the baseline experiment proposed by Hoy & Koulouri (2022), while introducing a novel generalization test via source-bias removal. The objective is to test multiple NLP pipelines in a unified framework under realistic distributional shifts.

## Pipeline Description

The script loads and preprocesses real and fake news articles from the ISOT dataset. It supports five feature extraction methods and six classifiers, tested under a standard stratified random split and a custom source-bias evaluation setting.

## Preprocessing

- Lowercasing, punctuation & URL removal
- Stopword filtering using NLTK
- Lemmatization via WordNet
- Minimal cleaning for contextual embeddings (Word2Vec, BERT)

## Feature Extraction Options

1. **Bag of Words** using `CountVectorizer` (max 1000 features)  
2. **TF-IDF** using `TfidfVectorizer` (max 1000 features)  
3. **Word2Vec (spaCy)**: average of token embeddings  
4. **DistilBERT**: transformer embeddings (Hugging Face)  
5. **Linguistic Cues**: sentence length, punctuation ratio, etc.

## Classifiers

Each representation is tested using six models from `sklearn`:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Neural Network (MLPClassifier)  
- AdaBoost  
- Gradient Boosting  
- Random Forest  

All evaluations use stratified 2-fold cross-validation.

## Generalization Test: Source-Bias Removal

A second evaluation simulates domain shift by training on:
- Fake news + a sample of true news
and testing on:
- Withheld true news (Reuters)

This allows assessing how robust each model is when source bias is removed from the training set.

## Outputs

- Per-model evaluation tables (Accuracy, Precision, Recall, F1)
- Bar plots comparing performance under random split vs. source-shift
- Exportable LaTeX tables and plots

## Configuration Parameters

- Sample reduction for faster runs:  
  `fake_df = fake_df.sample(n=2500)`  
  `real_df = real_df.sample(n=2500)`  

- TF-IDF and CountVectorizer limited to 1000 features  
- Stratified 2-fold CV used throughout  
- Random seeds fixed for reproducibility

## Dependencies

- `scikit-learn`  
- `spaCy` + `en_core_web_sm`  
- `transformers`  
- `nltk`  
- `pandas`, `matplotlib`, `seaborn`  



