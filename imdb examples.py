# imdb_examples.py
"""
LoRA fine-tuning example: DistilBERT sequence classifier on IMDb.

This shows:
- how to use HuggingFace Trainer
- how to wrap the base model with PEFT LoRA
- how to evaluate on IMDb with accuracy / F1

Run:
    python imdb_examples.py
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

from peft import LoraConfig, get_peft_model, TaskType

from data_utils import prepare_imdb_for_classification


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def main():
    base_model_name = "distilbert-base-uncased"

    # 1) tokenizer + 数据
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_ds = prepare_imdb_for_classification(tokenizer, split="train")
    test_ds = prepare_imdb_for_classification(tokenizer, split="test")

    # 2) 加载基础分类模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
    )

    # 3) 配置 LoRA
    # DistilBERT 里的自注意力层名字通常是 q_lin / v_lin
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_lin", "v_lin"],  # 只在注意力投影上加 LoRA
    )

    # 4) 把基础模型包成 LoRA 模型
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # 你会看到只有一小部分参数是可训练的

    # 5) TrainingArguments
    args = TrainingArguments(
        output_dir="./outputs/imdb_lora_classifier",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        learning_rate=2e-4,  # LoRA 可以适当学得更快一点
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        fp16=True,
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds.shuffle(seed=42).select(range(5000)),  # 取一部分做 demo
        eval_dataset=test_ds.shuffle(seed=42).select(range(2000)),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 7) 训练 + 评估
    trainer.train()
    metrics = trainer.evaluate()
    print("Eval metrics (LoRA DistilBERT on IMDb):", metrics)

    # 8) 保存 LoRA 适配器权重
    trainer.save_model("./outputs/imdb_lora_classifier")
    tokenizer.save_pretrained("./outputs/imdb_lora_classifier")


if __name__ == "__main__":
    main()
