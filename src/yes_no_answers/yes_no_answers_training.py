import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup, BertTokenizer, RobertaForSequenceClassification, DebertaForSequenceClassification
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json
import wandb  
import optuna  
from optuna.trial import TrialState
import sys
import os
from torch.optim.lr_scheduler import OneCycleLR
import random
from yes_no_answers.yes_no_answers_dataset import YesNoDataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class YesNoBertTrainer:
    def __init__(self, train_data, val_data, trial, device=None):
        set_seed()
        self.batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16])
        self.lr = trial.suggest_loguniform('lr', 5e-6, 5e-5)
        self.num_epochs = trial.suggest_int('num_epochs', 3, 10)
        self.max_length = trial.suggest_categorical('max_length', [128, 256, 512])
        dropout_prob = trial.suggest_uniform('dropout', 0.1, 0.5)
        weight_decay = trial.suggest_uniform('weight_decay', 0.0, 1e-4)
        warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.3)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = self.oversample_data(train_data)

        train_dataset = YesNoDataset(train_data, max_length=self.max_length)
        val_dataset = YesNoDataset(val_data, max_length=self.max_length)

        """self.model = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base',
            num_labels=2,
            hidden_dropout_prob=dropout_prob
        ).to(self.device)"""

        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=2,
            hidden_dropout_prob=dropout_prob
        ).to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.scaler = GradScaler()

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True  
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.trial_params = {
            'batch_size': self.batch_size,
            'learning_rate': self.lr,
            'num_epochs': self.num_epochs,
            'max_length': self.max_length,
            'dropout_prob': dropout_prob,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
        }

    def oversample_data(self, data):
        """
        Oversamples the minority class in the dataset.

        Args:
            data (list): The list of data entries (dictionaries).

        Returns:
            list: The oversampled data entries.
        """
        random.seed(42)

        data_by_class = {0: [], 1: []}
        for entry in data:
            yes_no_answer = entry['annotations'][0]['yes_no_answer']
            label = 1 if yes_no_answer == 'YES' else 0
            data_by_class[label].append(entry)

        class_counts = {label: len(entries) for label, entries in data_by_class.items()}
        print(f"Original class counts: {class_counts}")

        max_count = max(class_counts.values())

        # Oversample minority classes
        for label in data_by_class:
            entries = data_by_class[label]
            if len(entries) < max_count:
                oversampled_entries = random.choices(entries, k=max_count - len(entries))
                data_by_class[label].extend(oversampled_entries)

        oversampled_data = data_by_class[0] + data_by_class[1]

        random.shuffle(oversampled_data)

        new_class_counts = {
            label: len(entries) for label, entries in data_by_class.items()
        }
        print(f"New class counts after oversampling: {new_class_counts}")

        return oversampled_data

    def train(self):
        best_val_loss = float('inf')
        best_val_accuracy_at_best_loss = 0.0
        best_val_f1_at_best_loss = 0.0
        best_epoch = 0

        epochs_no_improve = 0
        n_epochs_stop = 2  

        wandb.init(
            project='yes_no_question_answering',  
            config=self.trial_params
        )

        wandb.watch(self.model, log='all', log_freq=100)

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for step, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()

                epoch_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

            avg_loss = epoch_loss / len(self.train_loader)
            train_accuracy = correct_predictions / total_predictions

            val_loss, val_accuracy, val_f1 = self.evaluate()

            wandb.log({
                'epoch': epoch + 1,
                'Training Loss': avg_loss,
                'Training Accuracy': train_accuracy,
                'Validation Loss': val_loss,
                'Validation Accuracy': val_accuracy,
                'Validation F1': val_f1,
                'Learning Rate': self.optimizer.param_groups[0]['lr'],
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_accuracy_at_best_loss = val_accuracy
                best_val_f1_at_best_loss = val_f1
                best_epoch = epoch + 1
                epochs_no_improve = 0

                torch.save(self.model.state_dict(), 'best_model.pth')

                wandb.log({
                    'Best Validation Loss': best_val_loss,
                    'Best Validation Accuracy': best_val_accuracy_at_best_loss,
                    'Best Validation F1': best_val_f1_at_best_loss,
                    'Best Epoch': best_epoch
                })
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_stop:
                    print(f"Early stopping triggered after {n_epochs_stop} epochs with no improvement.")
                    break

        wandb.run.summary['Best Validation Loss'] = best_val_loss
        wandb.run.summary['Best Validation Accuracy'] = best_val_accuracy_at_best_loss
        wandb.run.summary['Best Validation F1'] = best_val_f1_at_best_loss
        wandb.run.summary['Best Epoch'] = best_epoch

        wandb.finish()

        return best_val_loss, best_val_accuracy_at_best_loss, best_val_f1_at_best_loss

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                epoch_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions

        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        return avg_loss, accuracy, f1

def objective(trial):
    train_file = 'simplified-yes-no-train.jsonl'
    val_file = 'simplified-yes-no-dev.jsonl'

    with open(train_file, 'r') as f:
        train_data = [json.loads(line.strip()) for line in f if line.strip()]

    with open(val_file, 'r') as f:
        val_data = [json.loads(line.strip()) for line in f if line.strip()]

    # Initialize trainer with data and trial
    trainer = YesNoBertTrainer(
        train_data=train_data,
        val_data=val_data,
        trial=trial
    )

    best_val_loss, best_val_accuracy, best_val_f1 = trainer.train()

    trial.set_user_attr('best_val_accuracy', best_val_accuracy)
    trial.set_user_attr('best_val_f1', best_val_f1)

    return best_val_loss


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method('spawn', True)

    set_seed(42)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Best Validation Loss): {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_accuracy = trial.user_attrs.get('best_val_accuracy', None)
    best_f1 = trial.user_attrs.get('best_val_f1', None)

    if best_accuracy is not None:
        print(f"  Validation Accuracy at Best Loss: {best_accuracy:.4f}")
    if best_f1 is not None:
        print(f"  Validation F1 at Best Loss: {best_f1:.4f}")


