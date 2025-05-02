import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load IMDb dataset
dataset = load_dataset("imdb")

# Extract texts and labels
texts = dataset['train']['text'] + dataset['test']['text']
labels = dataset['train']['label'] + dataset['test']['label']

# Show basic label distribution
print("Overall label distribution:", Counter(labels))

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
features = vectorizer.fit_transform(texts)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Train label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))

# Logistic Regression Model
model = LogisticRegression(
    solver='liblinear',
    penalty='l2',
    C=1.0,                # Regularization strength
    max_iter=200,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
