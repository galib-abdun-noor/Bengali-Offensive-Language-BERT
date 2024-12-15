# Transliterated Bengali Offensive Language Classification

## Project Overview

This project addresses the challenge of detecting offensive language in transliterated Bengali (Banglish) text using both traditional machine learning models and advanced transformer-based models. The goal is to classify offensive and non-offensive content from a dataset of transliterated Bengali comments, leveraging the TB-OLID dataset as a benchmark for code-mixed text processing.

We evaluate both traditional machine learning (ML) models such as Logistic Regression and Random Forest and transformer-based models, including BanglaBERT, BanglishBERT, mBERT, XLM-RoBERTa, and others. The fine-tuned transformer models significantly outperform traditional models, with BanglaBERT achieving an F1-score of 76.98%.

## Key Features

- **TB-OLID Dataset**: A dataset of 5,000 annotated comments containing transliterated Bengali (Banglish) text, which poses challenges due to code-mixing and transliteration inconsistencies.
- **Traditional ML Models**: Includes Logistic Regression, Random Forest, SVM, and Naive Bayes, which utilize TF-IDF and Count Vectorization for feature extraction.
- **Transformer-based Models**: Utilizes models like BanglaBERT and BanglishBERT, fine-tuned specifically for offensive language detection in code-mixed Bengali text.
- **Performance Evaluation**: Evaluates models based on accuracy, precision, recall, and F1-score.

## Dataset

The TB-OLID dataset contains 5,000 Facebook comments in transliterated Bangla, labeled as offensive or non-offensive, with subcategories for targeted and untargeted offenses. The dataset is balanced with offensive and non-offensive comments and includes code-mixed and transliterated text challenges. The dataset is required for training and testing the models. You can download the TB-OLID dataset from the following link: [TB-OLID Dataset](https://github.com/mraihan-gmu/TB-OLID) Make sure the dataset is placed in the correct directory (./data).

## Project Setup

### Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Google Colab for GPU acceleration
