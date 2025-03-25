import wandb

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup

from datasets import load_dataset

import torch
import torch.nn as nn

import random
import numpy as np
import os
import gc
from tqdm import tqdm

wandb.login(key="c0a6a9a9deb94c8e5a185e86001c1de784e18b12")

SEED = 391
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def report_gpu():
    torch.cuda.empty_cache()
    gc.collect()

def adjust_labels(example):
    example["labels"] = int(example["labels"]) - 1  # Adjust if labels are 1-indexed
    return example

def tokenize_dataset(dataset, tokenizer):
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], max_length=256, padding='max_length', truncation=True),
        batched=True
    )
    dataset = dataset.rename_column('label', 'labels')  # Ensure 'label' column is correctly named
    dataset = dataset.map(adjust_labels)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset


def train(model, train_loader, eval_loader, run, optim, scheduler, config):

    os.makedirs('trained_models', exist_ok=True)

    batch_count = 0
    rounds_count = 0
    best_mae = float('inf')  # Lower MAE is better

    for epoch in range(config['num_epochs']):
        model.train()
        loop = tqdm(train_loader, leave=True)

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss

            optim.zero_grad()

            loss.backward()
            optim.step()

            scheduler.step()

            run.log({"train/loss": loss})

            # report_gpu()
            if batch_count % 200 == 0:
                model.eval()
                eval_loss = 0
                correct_preds = 0
                total_samples = 0
                total_abs_error = 0  # For MAE

                with torch.no_grad():
                    loop = tqdm(eval_loader, leave=True)
                    for eval_batch in loop:
                        # with torch.cuda.amp.autocast():
                        input_ids = eval_batch['input_ids'].to(device)
                        attention_mask = eval_batch['attention_mask'].to(device)
                        labels = eval_batch['labels'].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        eval_loss += outputs.loss.item()

                        preds = torch.argmax(outputs.logits, dim=-1)
                        correct_preds += (preds == labels).sum().item()

                        abs_error = torch.abs(preds - labels)
                        total_abs_error += abs_error.sum().item()

                        total_samples += labels.size(0)
                        # report_gpu()

                    avg_eval_loss = eval_loss / len(eval_loader)
                    accuracy = correct_preds / total_samples
                    mae = total_abs_error / total_samples

                    print("Eval loss: {:.3f}".format(avg_eval_loss))
                    print("Eval accuracy: {:.3f}".format(accuracy))
                    print("Eval MAE: {:.3f}".format(mae))

                    run.log({"eval/loss": avg_eval_loss})
                    run.log({"eval/accuracy": accuracy})
                    run.log({"eval/MAE": mae})

                    if mae < best_mae:
                        best_mae = mae
                        rounds_count = 0
                        try:
                            if batch_count > 0:
                                model.save_pretrained(f"trained_models/model_{run.get_url().split('/')[-1][4:]}_{epoch}_{batch_count}", from_pt=True)
                        except Exception as e:
                            print(f"Model not saved at epoch {epoch}, batch {batch_count}: {e}")

                    else:
                        rounds_count += 1

                    if rounds_count == config['early_stopping']:
                        print(f"Early stopping, model not improve WSD for {config['early_stopping']}")
                        return

                model.train()
            batch_count += 1
            loop.set_description(f'Epoch {epoch}')

        try:
            model.save_pretrained(f"trained_models/model_{run.get_url().split('/')[-1][4:]}_{epoch}", from_pt=True)
        except Exception as e:
            print(f'model not saved epoch = {epoch}, batch = {batch_count}: {e}')
        batch_count = 0

def verify_labels(dataset):
    unique_labels = set(dataset['train']['labels'].tolist())

    print(f"Unique labels in train: {unique_labels}")


if __name__ == "__main__":
    config = {"batch_size": 64,
              # "scale": 20.0,
              "learning_rate": 2e-6,
              "num_epochs": 10,
              "early_stopping": 50,
              # "reinit_n_layers": 3,
              "NUM_ACCUMULATION_STEPS": 8,
              "model": "youscan/ukr-roberta-base"} # 'xlm-roberta-base'

    run = wandb.init(project="review-sbert",
                     config=config)

    dataset = load_dataset('csv', data_files={'train': "cross_domain_uk_reviews/train_reviews.csv",
                                              'eval': "cross_domain_uk_reviews/eval_reviews.csv",
                                              'test': "cross_domain_uk_reviews/test_reviews.csv"})

    tokenizer = AutoTokenizer.from_pretrained(config['model'])

    dataset = tokenize_dataset(dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(config['model'], num_labels=5)
    # model = AutoModelForMaskedLM.from_pretrained("youscan/ukr-roberta-base")   

    model.config.problem_type = "single_label_classification"

    # model = nn.DataParallel(model)
    model.to(device)

    train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=config['batch_size'], shuffle=True, num_workers=8,
                                               pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset["eval"], batch_size=config['batch_size'], shuffle=True, num_workers=8,
                                              pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps - warmup_steps
                                                )

    train(model, train_loader, eval_loader, run, optim, scheduler, config)

    run.finish()
