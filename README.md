# 🎭 Sentiment Analysis on IMDb Reviews

This project performs sentiment analysis on IMDb movie reviews using three models:
- 🔹 Logistic Regression (baseline)
- 🔹 BERT (transformer-based baseline)
- 🔹 RoBERTa (fine-tuned for highest performance)

Built with Python, Hugging Face Transformers, and deployed with Flask for real-time predictions.

---

## 🧠 Overview

Sentiment analysis is the task of classifying the polarity (positive or negative) of text data.  
This project:
- Loads the IMDb dataset from Hugging Face.
- Compares three models: Logistic Regression, BERT, and RoBERTa.
- Fine-tunes the RoBERTa model to achieve 95%+ accuracy.
- Deploys the final RoBERTa model using Flask for web-based predictions.

---

## ⚙️ Tech Stack

- **Languages:** Python
- **Models:** Logistic Regression (scikit-learn), BERT, RoBERTa (Hugging Face Transformers)
- **Frameworks:** PyTorch, Flask
- **Data:** IMDb Movie Review Dataset (`datasets` library)

---
