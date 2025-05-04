# Report: Fake News Detection Using ISOT Dataset

## Aabstract

This study replicates the fake news detection baseline introduced by Hoy and Koulouri (2022) using the ISOT dataset. Five distinct text representation pipelines—Bag of Words, TF-IDF, Word2Vec (via spaCy), DistilBERT, and Linguistic Cues—are evaluated across six supervised classification algorithms under a stratified 2-fold cross-validation framework. The replication results closely mirror the original benchmarks, with ensemble classifiers leveraging TF-IDF and Bag of Words features achieving F1-scores consistently above 0.99. To assess model generalization beyond in-domain conditions, a novel source-bias removal experiment is introduced, simulating distributional shifts between training and testing sets. This experimental setup reveals notable performance degradation in most pipelines, underscoring the presence of source-specific biases within the dataset. The findings reinforce the importance of evaluating fake news detection systems under conditions that more accurately reflect real-world deployment scenarios.

## Dataset

- **ISOT Fake News Dataset**  
  - **Real news** from Reuters.com  
  - **Fake news** from unreliable news sources  
  - Preprocessed to remove stopwords, punctuation, and perform lemmatization  
  - Only the `text` field was used (no titles, dates, or subjects)

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


## Conclusion

- Shallow pipelines (TF-IDF, BoW) with ensemble models are strong in-domain.
- Deep models (BERT, Word2Vec) generalize better under source shifts.
- Linguistic Cues are less reliable when used in isolation.
- Generalization requires **source-debiased** evaluation setups.
