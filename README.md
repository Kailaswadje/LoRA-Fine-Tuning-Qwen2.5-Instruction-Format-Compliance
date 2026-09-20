# 🔧 LoRA Fine-Tuning on Qwen2.5 — Teaching a Chat Model a Strict Output Format

The sequel to my `distilgpt2` LoRA project: fine-tuning **Qwen2.5-0.5B-Instruct** — a real, modern, instruction-tuned chat model — using LoRA to enforce a **strict, structured response format** (`HR_RESPONSE: <answer>`) on top of correct HR knowledge. Where the first project taught a base model new facts, this one tests something harder: can LoRA teach an already-capable model a specific *output contract*, using the proper chat-template pipeline a production chat model actually expects?

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-fp16%20%2B%20CUDA-EE4C2C?logo=pytorch&logoColor=white)
![Qwen2.5](https://img.shields.io/badge/Qwen2.5--0.5B--Instruct-Base%20Model-6236FF)
![PEFT](https://img.shields.io/badge/🤗%20PEFT-LoRA-FFD21E)

---

## 📌 Overview

This project is a deliberate step up from an earlier `distilgpt2` LoRA fine-tuning notebook in the same series — same core technique, harder target. Instead of a small base model with no chat awareness, this notebook fine-tunes **Qwen2.5-0.5B-Instruct**, a real instruction-tuned model that already understands chat-style prompting via `tokenizer.apply_chat_template()`. The fine-tuning goal is also different: not just correct answers, but answers in an **exact, enforced format** — `HR_RESPONSE: <your response>` — every single time.

---

## 🏗️ The Complete Pipeline

```
Qwen/Qwen2.5-0.5B-Instruct (pretrained, frozen, fp16, on CUDA)
        │
        ▼
BASELINE test (via proper chat template) ──► inconsistent format, no HR_RESPONSE: prefix
        │
        ▼
Instruction-formatted training examples ──► tokenized dataset
        │
        ▼
LoraConfig (r=8, alpha=16, target_modules=["q_proj","k_proj","v_proj","o_proj"])
        │
        ▼
get_peft_model() ──► injects LoRA adapters into ALL FOUR attention projections
        │
        ▼
model.print_trainable_parameters()
        │
        ▼
Trainer.train() ──► 15 epochs, fp16, only adapter weights update
        │
        ▼
AFTER test ──► consistent "HR_RESPONSE:" formatted, correct answers
        │
        ▼
Tested again on a SECOND unseen-format question
```

---

## 🔬 Part 1 — Instruction-Formatted Training Data (Not Just Q&A)

```python
{"text": """### Instruction:
Respond to the employee using exactly this format:
HR_RESPONSE: <your response>

### Employee:
I need leave tomorrow.

### Response:
HR_RESPONSE: Please submit your leave request through the HR portal."""}
```
Unlike the earlier `distilgpt2` project's plain `Question:/Answer:` pairs, every training example here is a **full instruction-tuning-style block**: an explicit instruction, the employee's message, and the required response format. This mirrors how real instruction-tuned models are actually trained — teaching format compliance, not just factual recall.

---

## 🔬 Part 2 — Loading Qwen2.5 in Half Precision on GPU

```python
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to("cuda")
```
`torch_dtype=torch.float16` loads the model in **half precision**, roughly halving GPU memory usage compared to full 32-bit floats — a real, necessary optimization once working with an actual instruction-tuned model rather than a tiny demo model. This notebook has a **hard GPU dependency**; there is no CPU fallback path.

---

## 🔬 Part 3 — Prompting the Proper Way: Chat Templates

```python
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
```
Rather than feeding a raw prompt string directly to the tokenizer (as the `distilgpt2` project did), this notebook uses `apply_chat_template()` — the correct way to prompt any instruction-tuned model, since it wraps the message in whatever special tokens and role markers that specific model was trained to expect. Skipping this step is a common, silent source of degraded chat-model output.

---

## 🔬 Part 4 — LoRA Targeting FOUR Attention Projections, Not One

```python
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM"
)
```
This is the clearest proof of the "architecture-specific" lesson from the earlier LoRA project: GPT-2's single fused `c_attn` layer has no equivalent here. Qwen2.5, like most modern transformer architectures, splits attention into **four separate projections** — query, key, value, and output — and all four are targeted for LoRA adapters here, a direct architectural contrast worth understanding before configuring LoRA on any new model family.

---

## 🔬 Part 5 — Training, Then a Real Before/After Comparison

```python
training_args = TrainingArguments(
    output_dir="./qwen_lora_hr", per_device_train_batch_size=1,
    num_train_epochs=15, learning_rate=2e-4, fp16=True
)
trainer.train()
```
Fifteen epochs — fewer than the `distilgpt2` project's thirty — reflecting that an instruction-tuned model already starts with far stronger format-following instincts; LoRA here is refining existing behavior, not building it from nothing.

**Before:**
```
[inconsistent output, no enforced HR_RESPONSE: prefix]
```
**After:**
```
HR_RESPONSE: Please submit your leave request through the HR portal.
```
A **second, unseen-format question** ("Can I work from home?") is also tested after training, checking whether the format constraint generalizes beyond the exact examples the model was trained on — not just memorized verbatim.

---

## 🗂️ Repository Structure

```
lora-fine-tuning-qwen2-5/
├── GenAI_28_LORA_WITH_QWENN_MODEL.ipynb   # Main notebook
├── requirements.txt                         # Dependencies
├── .gitignore                               # Keeps model cache, checkpoints & secrets out of git
├── .env.example                             # Template for optional environment variables
└── README.md                                # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- **A CUDA-capable GPU is required** — the notebook hardcodes `.to("cuda")` with no CPU fallback (Google Colab's free GPU tier is sufficient for this model size)

### Installation

```bash
git clone https://github.com/Kailaswadje/lora-fine-tuning-qwen2-5.git
cd lora-fine-tuning-qwen2-5

pip install -r requirements.txt

jupyter notebook GenAI_28_LORA_WITH_QWENN_MODEL.ipynb
```

> 💡 No API key required — `Qwen/Qwen2.5-0.5B-Instruct` is a public model that downloads directly from Hugging Face Hub without authentication. If you hit a `torchao` version conflict during install, the notebook includes an explicit uninstall/reinstall step (`torchao>=0.16.0`) that resolves it.

---

## 🧠 Key Takeaways

- **`target_modules` truly is architecture-specific** — GPT-2's single `c_attn` versus Qwen's four separate `q_proj`/`k_proj`/`v_proj`/`o_proj` layers is the concrete proof, not just a warning in documentation
- **Chat templates matter** — `apply_chat_template()` is the correct prompting interface for any instruction-tuned model, and skipping it silently degrades output quality regardless of how good the fine-tuning was
- **LoRA can teach format compliance, not just facts** — enforcing an exact `HR_RESPONSE:` prefix is a genuinely different (and arguably harder) target than teaching new factual content
- **A stronger base model needs less fine-tuning** — fifteen epochs here versus thirty for `distilgpt2` reflects Qwen's existing instruction-following strength being refined, not built from scratch
- **fp16 + explicit CUDA placement is a real memory-management decision** — necessary once working with an actual chat-capable model rather than a toy-scale demo

---

## 📚 Series Context

| Project | Base Model | What LoRA Teaches |
|---|---|---|
| [LoRA Fine-Tuning on distilgpt2](https://github.com/Kailaswadje/lora-fine-tuning-distilgpt2) | `distilgpt2` (no chat awareness) | New factual knowledge |
| **This repo** | `Qwen2.5-0.5B-Instruct` (already instruction-tuned) | Strict output format compliance |

---

## 🔮 Possible Extensions

- [ ] Test format compliance across a wider variety of unseen employee questions to measure generalization more rigorously
- [ ] Compare LoRA adapter size and training time between the `distilgpt2` and Qwen2.5 projects directly
- [ ] Add a JSON-structured output format instead of a plain-text prefix, and evaluate parse success rate
- [ ] Merge the LoRA adapter into the base model weights (`merge_and_unload()`) and compare inference speed

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If this showed you LoRA's flexibility across model architectures, consider giving it a star!
