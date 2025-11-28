
# chatbot_example_3.py
"""
Simple CLI chatbot that uses the LoRA-finetuned causal LM.

Run:
    python chatbot_example_3.py
"""
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


def load_lora_model(lora_dir: str):
    lora_dir = Path(lora_dir)
    config = PeftConfig.from_pretrained(lora_dir)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def chat_loop(model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)

    print("🔮 LoRA Chatbot loaded. Type 'exit' to quit.")
    history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Chatbot: Goodbye!")
            break

        history.append(f"User: {user_input}")
        prompt = "\n".join(history) + "\nAssistant:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = full_text.split("Assistant:")[-1].strip()
        history.append(f"Assistant: {reply}")
        print(f"Chatbot: {reply}")


def main():
    lora_dir = "./outputs/lora_causal"
    model, tokenizer = load_lora_model(lora_dir)
    chat_loop(model, tokenizer)


if __name__ == "__main__":
    main()
