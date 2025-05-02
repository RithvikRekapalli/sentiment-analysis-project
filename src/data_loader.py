from datasets import load_dataset
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split

def load_and_tokenize_dataset():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")

    print("Tokenizing dataset...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize_fn(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    
    train_val = dataset["train"].train_test_split(test_size=0.05, seed=42) 

    tokenized_train = train_val["train"].map(tokenizer, batched=True)
    tokenized_val = train_val["test"].map(tokenizer, batched=True)
    tokenized_test = dataset["test"].map(tokenizer, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return {
        "train": tokenized_train,
        "val": tokenized_val,
        "test": tokenized_test
    }
