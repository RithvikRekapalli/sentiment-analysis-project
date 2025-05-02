import os
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_and_tokenize_dataset

def main():
    # Load and tokenize dataset
    tokenized_datasets = load_and_tokenize_dataset()

    # Load pre-trained RoBERTa model
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["val"],
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
