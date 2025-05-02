import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load IMDb dataset
dataset = load_dataset("imdb")

# Use smaller sample if needed: dataset["train"].select(range(1000))

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics function
def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, predictions)
    print("\nClassification Report:\n", classification_report(labels, predictions))
    return {"accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_baseline_results",
    evaluation_strategy="epoch",
    save_strategy="no",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(10000)),
    eval_dataset=tokenized_datasets["test"].select(range(2000)),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train and evaluate
trainer.train()
trainer.evaluate()