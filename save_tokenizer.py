from transformers import RobertaTokenizer

# Load tokenizer from base pretrained model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Save tokenizer files to the checkpoint directory
tokenizer.save_pretrained("results/checkpoint-6000")
