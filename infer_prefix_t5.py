
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # HF
from peft import PeftModel                                     # HF PEFT


def main():
    base_model_name = "t5-small"
    adapter_path = "./outputs/peft_prefix_t5_summarization_adapter"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    article = (
        "The movie tells the story of a young woman who moves to a big city to pursue her dreams. "
        "Despite many challenges and setbacks, she finds friendship, love, and her own voice."
    )
    model_input = "summarize: " + article

    inputs = tokenizer(model_input, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=4,
        )

    print("Summary:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
