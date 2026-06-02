# NLP Text Classification — BBC News Dataset

**End-to-end NLP pipeline to classify BBC News articles into topic categories using Python, NLTK, and Scikit-learn.**

---

## What this does

- Preprocesses raw BBC News text data (tokenisation, stopword removal, lemmatisation)
- Engineers features using TF-IDF vectorisation
- Trains and evaluates multiple classification models (Naive Bayes, Logistic Regression, SVM)
- Compares model performance using accuracy, precision, recall, and F1 score

---

## Key findings

Preprocessing choices had more impact on classification accuracy than the choice of classifier. TF-IDF with sublinear term frequency scaling and bigrams consistently outperformed bag-of-words approaches across all classifiers tested.

---

## Stack

- Python (NLTK, Scikit-learn, Pandas, NumPy)
- Jupyter Notebook
- Matplotlib / Seaborn for visualisation

---

## Dataset

BBC News dataset — 2,225 articles across 5 categories: business, entertainment, politics, sport, tech.

---

## Files

| File | Description |
|------|-------------|
| `*.ipynb` | Main notebook — full pipeline from raw text to classification results |

---

## Skills demonstrated

`NLP` `Text Classification` `TF-IDF` `NLTK` `Scikit-learn` `Feature Engineering` `Model Evaluation` `Python`
