from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the trained model
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_chatbot")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_chatbot").to("cpu")  # Use CPU on Heroku

# Initialize Flask app
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, no_repeat_ngram_size=2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
