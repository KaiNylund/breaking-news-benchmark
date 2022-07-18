from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments, trainer_utils
from datasets import *
from torch import nn
import numpy as np
import torch
import wandb

NUM_EPOCHS = 1
TRAIN_SIZE = 100000000
TEST_SIZE = 5000000

wandb.login(key="c856468e9d543f69e0da2cae8e515ae88d412636")

#os.environ["WANDB_RUN_GROUP"] = "train-year-" + str(train_year) + wandb.util.generate_id()
wandb.init(project='mtg-gpt2-finetuning', entity="kainylund", config={"epochs": NUM_EPOCHS}, name="token-multi-training")

train_year_start = 2012
train_year_end = 2021
test_year_start = 2012
test_year_end = 2021

'''class MultiEvalMetrics:
    def __init__(self, dataset_sizes):
        self.dataset_sizes = dataset_sizes

    def compute_metrics(self, p):
        torch.cuda.empty_cache()
        metrics = {}
        loss = nn.CrossEntropyLoss()
        cumulative_sizes = self.dataset_sizes.copy()
        cumulative_sizes.insert(0, 0)
        cumulative_sizes = np.cumsum(np.array(cumulative_sizes))
        print(p)

        for i in range(1, len(cumulative_sizes)):
            dataset_start = cumulative_sizes[i - 1]
            dataset_end = cumulative_sizes[i]
            cur_name = str(year_start - 1 + i) + "-test-loss"
            print(cur_name)

            labels = p.label_ids[dataset_start:dataset_end]
            predictions = p.predictions[dataset_start:dataset_end, :]

            test_loss = loss(predictions, labels)
            metrics[cur_name] = test_loss

        wandb.log(metrics)
        return metrics
'''

def finetune_with_year(train_year):
    print("Starting to train with data from " + str(train_year) + " -----------------------------")
    torch.cuda.empty_cache()
    data_files = {"train": "./WMTdata/token_splits/tokens_" + str(train_year) + "_" + str(TRAIN_SIZE) + ".txt",
                  "blank": "./WMTdata/blank.txt"}

    for test_year in range(test_year_start, test_year_end + 1):
        data_files["test_" + str(test_year)] = "./WMTdata/token_splits/tokens_" + str(test_year) + "_" + str(TEST_SIZE) + ".txt"

    raw_datasets = load_dataset("text", data_files=data_files, cache_dir="./cache")
    tokenizer_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir="./cache")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.save_to_disk("./WMTdata/tokenized_datasets/")

    block_size = 1024
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of
        # this drop, you can customize this part to your needs.
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
        batch_size=1,
    )
    lm_datasets.save_to_disk("./WMTdata/lm_datasets/")

    #test_set_lens = []
    #test_sets = []
    #for test_year in range(year_start, year_end + 1):
    #    cur_test_set = lm_datasets["test_" + str(test_year)]
    #    test_set_lens.append(cur_test_set.num_rows)
    #    test_sets.append(cur_test_set)

    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

    args = TrainingArguments(
        output_dir="./models/" + str(train_year) + "/",
        #evaluation_strategy = "steps",
        evaluation_strategy = "epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs = NUM_EPOCHS,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        report_to="wandb"
        #logging_steps=1,
        #eval_steps=1
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["blank"]
    )

    trainer.train()
    trainer.save_model()
    #wandb.finish()

    train_year_metrics = {}
    for test_year in range(test_year_start, test_year_end + 1):
        metrics = trainer.evaluate(eval_dataset=lm_datasets["test_" + str(test_year)])
        train_year_metrics[test_year] = metrics
        print(metrics)

    np.save("./token_metrics/train-year-" + str(train_year), train_year_metrics)

for train_year in range(train_year_start, train_year_end + 1):
    finetune_with_year(train_year)