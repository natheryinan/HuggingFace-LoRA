# train_lora_causal.py
"""
LoRA fine-tuning for a causal language model (e.g., GPT-2) on IMDb text.
This is a compact but realistic PEFT + Transformers training script.
"""

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from data_utils import prepare_imdb_for_causal_lm, CausalLMCollator


def main():
    base_model_name = "gpt2"
    output_dir = Path("./outputs/lora_causal")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = prepare_imdb_for_causal_lm(tokenizer, split="train", max_length=128)

    model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],  # GPT-2 attention projections
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
    )

    data_collator = CausalLMCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds.shuffle(seed=42).select(range(2000)),  # small subset
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
