from datasets import load_dataset

sft_ds_name = 'CarperAI/openai_summarize_tldr'
sft_ds = load_dataset(sft_ds_name)
sft_train = sft_ds['train']
sft_valid = sft_ds['valid']
sft_test = sft_ds['test']

# Define data formatting function
def format_example(example):
    text = f"### Text: {example['prompt']}\n### Summary: {example['label']}"
    return text

# Demonstrate formatting
for example in sft_train:
    print(format_example(example))
    break

# Configure the model
import torch
from trl import ModelConfig, get_quantization_config, get_kbit_device_map
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer

model_config = ModelConfig(
    model_name_or_path='facebook/opt-350m'
)

torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)

tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set up the evaluation metric
import evaluate
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    if isinstance(eval_preds, tuple):
        eval_preds = eval_preds[0]
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    return result

# Configure the trainer
from trl import SFTTrainer
from transformers import TrainingArguments

num_epochs = 10
training_args = TrainingArguments(
    output_dir='./save_model',
    evaluation_strategy="epoch",
    save_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    adam_beta1=0.9,
    adam_beta2=0.95,
    num_train_epochs=num_epochs,
    load_best_model_at_end=True,
)

max_input_length = 512
trainer = SFTTrainer(
    model=model_config.model_name_or_path,
    model_init_kwargs=model_kwargs,
    args=training_args,
    train_dataset=sft_train,
    eval_dataset=sft_valid,
    max_seq_length=max_input_length,
    tokenizer=tokenizer,
    peft_config=peft_config,
    compute_metrics=compute_metrics,
    packing=True,
    formatting_func=format_example
)

trainer.train()
# Load the dataset
from datasets import load_dataset

rw_ds_name = 'CarperAI/openai_summarize_comparisons'
rw_ds = load_dataset(rw_ds_name)

# Preprocess the dataset
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen = f"### Text: {prompt}\n### Summary: {chosen}"
        tokenized_chosen = tokenizer(chosen)

        rejected = f"### Text: {prompt}\n### Summary: {rejected}"
        tokenized_rejected = tokenizer(rejected)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

rw_ds_processed = rw_ds.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

max_input_length = 512

rw_ds_filtered = rw_ds_processed.filter(
    lambda x: len(x["input_ids_chosen"]) <= max_input_length
    and len(x["input_ids_rejected"]) <= max_input_length
)

rw_train = rw_ds_filtered["train"]
rw_valid = rw_ds_filtered["valid1"]
rw_test = rw_ds_filtered["test"]

# Load the model
import torch
from trl import ModelConfig, get_quantization_config, get_kbit_device_map
from transformers import AutoModelForSequenceClassification

model_config = ModelConfig(
    model_name_or_path='facebook/opt-350m'
)

torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False,
    device_map=get_kbit_device_map() if quantization_config is not None else None,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_config.model_name_or_path, num_labels=1, **model_kwargs
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

# Train the reward model
from trl import RewardConfig, RewardTrainer

num_epochs = 1

reward_config = RewardConfig(
    output_dir='./save_rw_model',
    evaluation_strategy="epoch",
    save_strategy='epoch',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=num_epochs,
    load_best_model_at_end=True,
    max_length=max_input_length,
)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=rw_train,
    eval_dataset=rw_valid,
    peft_config=peft_config,
)

trainer.train()
from transformers import AutoTokenizer

# Load the dataset
ppo_ds_name = "CarperAI/openai_summarize_tldr"
ppo_ds = load_dataset(ppo_ds_name, split="train")

# Build the dataset
def build_dataset(ds, tokenizer, max_length=200):
    ds = ds.filter(lambda x: len(x["prompt"]) > max_length, batched=False)

    def tokenize(sample):
        sample["text"] = sample["prompt"] + sample["label"]
        sample["input_ids"] = tokenizer.encode(sample["text"])[:max_length]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
tokenizer.pad_token = tokenizer.eos_token
ppo_ds = build_dataset(ppo_ds, tokenizer)

# Load the model
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
)

model_path = "./save_sft_model/checkpoint-1000"
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    pretrained_model_name_or_path=model_path,
    peft_config=peft_config,
)

# Set up the trainer
from trl import PPOConfig, PPOTrainer

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

ppo_config = PPOConfig(
    model_name="facebook/opt-350m"
)

device = 0 if torch.cuda.is_available() else "cpu"
ppo_trainer = PPOTrainer(ppo_config, model, None, tokenizer, dataset=ppo_ds, data_collator=collator)

# Load the reward model
from transformers import AutoModelForSequenceClassification, pipeline

rw_model = AutoModelForSequenceClassification.from_pretrained("./save_rw_model")
sentiment_pipe = pipeline("sentiment-analysis", model=rw_model, device=device)

if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

if sentiment_pipe.model.config.pad_token_id is None:
    sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

# Train the model
from tqdm import tqdm

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 200,
}
sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}

for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the model
    response_tensors, ref_response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
    )
    batch["response"] = tokenizer.batch_decode(response_tensors)
    batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
    ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
    ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
    batch["ref_rewards"] = ref_rewards

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])