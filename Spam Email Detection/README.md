# Spam Email Detection

A machine learning project that classifies emails as spam or ham (not spam) using Natural Language Processing (NLP) and a Random Forest classifier.

## Overview

This project builds a text classification model to detect whether an email is spam or legitimate.

The workflow includes text preprocessing, feature extraction using Bag-of-Words, model training with Random Forest, and testing the model on an individual email.

## Dataset

The project uses a spam/ham email dataset containing **5,171 emails** with the following columns:

- `label` - Email category: ham or spam
- `text` - Email content
- `label_num` - Numeric representation of the label

The dataset contains **no missing values**.

## Workflow

### 1. Data Loading
- Load the email dataset using Pandas
- Inspect the dataset structure and dimensions
- Check for missing values

### 2. Text Preprocessing

The email text is processed using NLTK:

- Convert text to lowercase
- Remove punctuation
- Remove English stopwords
- Apply Porter Stemming
- Build a cleaned text corpus

### 3. Feature Extraction

The processed text is converted into numerical features using:

- `CountVectorizer`
- Bag-of-Words representation

The data is then split into:

- 80% training data
- 20% testing data

### 4. Machine Learning Model

A `RandomForestClassifier` is trained on the processed email data.

The model uses parallel processing with:

```python
RandomForestClassifier(n_jobs=-1)
