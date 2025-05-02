from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer
from datasets import load_dataset
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Load IMDb dataset
dataset = load_dataset("imdb")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

tokenized_test = dataset['test'].map(tokenize, batched=True)
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model from checkpoint
model = RobertaForSequenceClassification.from_pretrained("./results/checkpoint-6000")

# Use Trainer for prediction
trainer = Trainer(model=model)
predictions = trainer.predict(tokenized_test)

# Extract labels and predictions
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Compute metrics
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
accuracy = accuracy_score(labels, preds)

print(f"\n✅ Evaluation Metrics from checkpoint-6000:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
