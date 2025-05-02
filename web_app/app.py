from flask import Flask, render_template, request
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = Flask(__name__)

# Path to your model and tokenizer
model_path = "../results/checkpoint-6000"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None  # Define it initially as None
    if request.method == "POST":
        text = request.form["text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs).item()
        prediction = "Positive" if predicted_class == 1 else "Negative"
        if prediction == 'Positive':
            emoji = '😄'
        else:
            emoji = '😞'

    return render_template("index.html", prediction=prediction, emoji=emoji)


if __name__ == "__main__":
    app.run(debug=True)
