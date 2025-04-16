import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model from Hugging Face Hub
model_name = "muhtasimimam/mh-chatbot"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Set padding token (important for generation)
tokenizer.pad_token = tokenizer.eos_token

# Safety filter: flag dangerous intent and provide helpful response
def safe_generate(input_text):
    input_text_lower = input_text.lower()

    harmful_keywords = ["kill myself", "suicide", "end it", "want to die", "die", "hurt myself"]
    if any(kw in input_text_lower for kw in harmful_keywords):
        return "I'm really sorry you're feeling this way. You're not alone — please consider reaching out to a mental health professional or helpline in your area. You matter."

    # Encode input and generate response
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.strip()

# Gradio interface
iface = gr.Interface(
    fn=safe_generate,
    inputs=gr.Textbox(lines=3, placeholder="How are you feeling today?"),
    outputs="text",
    title="🧠 AI-Powered Mental Health Chatbot",
    description="This chatbot is here to listen. It's not a replacement for professional help, but it can talk with you if you're feeling down. 💬",
)

iface.launch()
