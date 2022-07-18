import numpy as np
#from tokenizers import ByteLevelBPETokenizer
from transformers import TextDataset, GPT2TokenizerFast, GPT2Model, AutoModelForCausalLM, AutoTokenizer
#from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import *
from timeout import timeout
import wandb

wandb.login(key=["c856468e9d543f69e0da2cae8e515ae88d412636"])

num_epochs = 5
year = 2013
# Toy example to train a BPE tokenizer
#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train(files=["./WMTdata/train_" + str(year) + ".txt"], vocab_size=52_000, min_frequency=3, special_tokens=[
#    "<s>",
#    "<pad>",
#    "</s>",
#    "<unk>",
#    "<mask>",
#])
#tokenizer.save_model("./models/", "WMT-" + str(year) + "-bpe-test")
#tokenizer.save_pretrained("./models/WMT-" + str(year) + "-bpe-test-pretrained")

#train_data = load_dataset('csv', data_files='./WMTdata/decoded_splits_2013.csv')
#test_data = load_dataset('csv', data_files='./WMTdata/decoded_splits_2018.csv')
#raw_datasets = DatasetDict({"train": from_dict(train_data), "test": from_dict(test_data)})

wandb.init(project='gpt2-finetuning-' + str(year), entity="kainylund", config={"epochs": num_epochs})

data_files = {"train": "./WMTdata/text_2016.txt", "test": "./WMTdata/text_" + str(year) + ".txt"}
raw_datasets = load_dataset("text", data_files=data_files)
print(raw_datasets)

model_checkpoint = "gpt2"
#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
#tokenizer.pad_token = tokenizer.eos_token
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

#context_length = 128
#def tokenize_function(element):
#    outputs = tokenizer(
#        element["text"],
#        truncation=True,
#        max_length=context_length,
#        return_overflowing_tokens=True,
#        return_length=True,
#    )
#    input_batch = []
#    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#        if length == context_length:
#            input_batch.append(input_ids)
#    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
tokenized_datasets.save_to_disk("./WMTdata")

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

#config = AutoConfig.from_pretrained(
#    "gpt2",
#    vocab_size=len(tokenizer),
#    n_ctx=context_length,
#    bos_token_id=tokenizer.bos_token_id,
#    eos_token_id=tokenizer.eos_token_id,
#)

#model = GPT2LMHeadModel(config)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

#args = TrainingArguments(
#    output_dir="models",
#    per_device_train_batch_size=32,
#    per_device_eval_batch_size=32,
#    evaluation_strategy="steps",
#    eval_steps=5_000,
#    logging_steps=5_000,
#    gradient_accumulation_steps=8,
#    num_train_epochs=1,
#    weight_decay=0.1,
#    warmup_steps=1_000,
#    lr_scheduler_type="cosine",
#    learning_rate=5e-4,
#    save_steps=5_000,
#)

args = TrainingArguments(
    output_dir="./models/",
    evaluation_strategy = "steps",
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs = num_epochs
)

#metric = load_metric("accuracy")
#def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return metric.compute(predictions=predictions, references=labels)

#print(lm_datasets)
#print(lm_datasets["train"]["input_ids"][:100])
#print(lm_datasets["train"]["labels"][:100])

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"]
)

wandb.log({"loss": loss})

trainer.train()
trainer.save_model()