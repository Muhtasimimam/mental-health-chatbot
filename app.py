from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, os

# Load model & tokenizer from Hugging Face Hub
REPO_ID   = "muhtasimimam/mh-chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(REPO_ID, token=os.environ.get("HF_TOKEN"))
model     = GPT2LMHeadModel.from_pretrained(REPO_ID,    token=os.environ.get("HF_TOKEN")).to("cpu")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    text = data.get("message", "")
    inputs  = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, no_repeat_ngram_size=2)
    reply   = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({"response": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
