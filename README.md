\# Hugging Face PEFT Lab · LoRA + Prefix Tuning + Mini-RLHF

This project is my personal playground for **making large language models behave** —  
from classic fine-tuning to **LoRA**, **prefix tuning**, and a **toy RLHF loop** on top of IMDb sentiment.

## What I Built

- **LoRA fine-tuning for GPT-2 (Causal LM)** on IMDb-style text to power a small chatbot
- **LoRA fine-tuning for DistilBERT (Sequence Classification)** for IMDb sentiment analysis
- **Prefix-tuning T5** for summarization-style tasks
- A **CLI chatbot** that runs on top of the LoRA GPT-2 policy
- A planned **mini-RLHF loop**:
  - SFT on (prompt, answer) sentiment explanation pairs
  - Reward signal based on correctness + explanation quality
  - Policy update with PPO / policy-gradient on top of the LoRA adapter

The goal is not to “rebuild ChatGPT”, but to **rebuild the core ideas** in a smaller, inspectable setting.

---

## Architecture at a Glance

### 1. Supervised Fine-Tuning (SFT)
- Start from a pretrained base model: GPT-2 / DistilBERT / T5
- Use Hugging Face Trainer + PEFT/LoRA to adapt to:
  - IMDb sentiment **classification** (DistilBERT + LoRA)
  - IMDb-style **generation / opinions** (GPT-2 + LoRA)
  - Short **summaries** (T5 + prefix-tuning)

### 2. Reward Modeling (planned)
- For a given sentiment prompt, generate multiple answers
- Score answers using:
  - Sentiment correctness (positive/negative)
  - Presence of a natural-language explanation
  - Length / style penalties
- Train or approximate a **Reward Model** that estimates:  
  `R(prompt, answer) → scalar reward`

### 3. RLHF-style Policy Update (planned)
- Treat the LoRA-adapted GPT-2 as the **policy**
- Use PPO or a simple policy-gradient step to:
  - Sample answer
  - Compute reward
  - Update policy parameters (only LoRA weights)  
- Keep a reference policy to avoid drifting too far from the SFT behavior

---

## Files

- `data_utils.py` – dataset loading, tokenization, and collators for IMDb
- `imdb_examples.py` – DistilBERT + LoRA sentiment classifier
- `train_lora_causal.py` – GPT-2 + LoRA training on IMDb text
- `chatbot_example_3.py` – multi-turn CLI chatbot using the LoRA GPT-2 policy
- `infer_prefix_t5.py` – T5 / prefix-tuning summarization inference
- `infrastructure.ini` – model / hyperparameter configuration
- `requirements.txt` – dependencies

---

## Why This Matters for LLM Roles

This lab shows that I can:

- Move beyond “call an API” and actually **shape model behavior**
- Implement **PEFT / LoRA / prefix-tuning** on top of Hugging Face
- Understand the **difference between classification and causal LM setups**
- Design the core loop of **RLHF** (SFT → RM → policy update)
- Think in terms of **adapters, personas, and domain-specialist experts**

It’s a small repo, but it encodes the **same ideas that power modern systems like ChatGPT and Gemini—just in a form that one person can understand end-to-end.**
