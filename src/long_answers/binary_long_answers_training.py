import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import DebertaForSequenceClassification, get_linear_schedule_with_warmup, BertForSequenceClassification, RobertaForSequenceClassification
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import wandb
import optuna
import random
import pickle
from torch.utils.data import TensorDataset

count = -1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LongAnswerTrainer:
    def __init__(self, train_data, val_data, trial, device=None):
        set_seed()
        self.trial = trial

        self.batch_size = trial.suggest_categorical('batch_size', [16])
        self.lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
        self.num_epochs = trial.suggest_int('num_epochs', 2, 8)
        dropout_prob = trial.suggest_float('dropout', 0.1, 0.3)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.01)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.0, 0.3)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=96)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=96)

        """self.model = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base',
            num_labels=1,  # Binary classification
            hidden_dropout_prob=dropout_prob
        ).to(self.device, non_blocking=True)"""

        """self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=1,  # Binary classification
            hidden_dropout_prob=dropout_prob
        ).to(self.device, non_blocking=True)"""

        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=1,  
            hidden_dropout_prob=dropout_prob
        ).to(self.device, non_blocking=True)

        self.criterion = torch.nn.BCEWithLogitsLoss()

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        total_steps = len(self.train_loader) * self.num_epochs
        self.num_training_steps = total_steps
        self.num_warmup_steps = int(total_steps * warmup_ratio)

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps)

        self.scaler = torch.amp.GradScaler()

    def train(self):
        best_f1 = 0.0
        best_epoch = 0
        global count
        count += 1

        wandb.init(project="long_answer_question_answering", config={
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.num_epochs,
            "dropout": self.model.config.hidden_dropout_prob,
            "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
            "warmup_ratio": self.num_warmup_steps / self.num_training_steps
        })

        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input_ids, attention_mask, token_type_ids, labels = [b.to(self.device, non_blocking=True) for b in batch]

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )
                    logits = outputs.logits.squeeze(-1) 
                    loss = self.criterion(logits, labels.float())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            val_loss, f1, precision, recall, accuracy = self.evaluate()

            wandb.log({
                "Training Loss": avg_loss,
                "Validation Loss": val_loss,
                "Validation F1": f1,
                "Validation Precision": precision,
                "Validation Recall": recall,
                "Validation Accuracy": accuracy,
                "Epoch": epoch + 1
            })

            print(f"\nEpoch {epoch + 1}/{self.num_epochs} Summary:")
            print(f"  Training Loss: {avg_loss:.4f}")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation F1: {f1:.4f}")
            print(f"  Validation Precision: {precision:.4f}")
            print(f"  Validation Recall: {recall:.4f}")
            print(f"  Validation Accuracy: {accuracy:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), f"best_model_trial_{count}.pth")
                print(f"  Best model saved at epoch {best_epoch} with F1: {best_f1:.4f}")

        wandb.finish()
        print(f"\nTraining complete. Best F1 score: {best_f1:.4f} at epoch {best_epoch}.")
        return best_f1

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, attention_mask, token_type_ids, labels = [b.to(self.device, non_blocking=True) for b in batch]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits = outputs.logits.squeeze(-1)  
                loss = self.criterion(logits, labels.float())
                total_loss += loss.item()

                preds = torch.sigmoid(logits).cpu().numpy()
                preds = (preds >= 0.5).astype(int)
                labels = labels.cpu().numpy()

                all_preds.extend(preds.flatten())
                all_labels.extend(labels.flatten())

        avg_loss = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, f1, precision, recall, accuracy

def load_data_from_pickle(train_file, val_file):
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)

    train_input_ids = torch.tensor([entry['input_ids'] for entry in train_data], dtype=torch.long)
    train_attention_mask = torch.tensor([entry['attention_mask'] for entry in train_data], dtype=torch.long)
    train_token_type_ids = torch.tensor([entry['token_type_ids'] for entry in train_data], dtype=torch.long)
    train_labels = torch.tensor([entry['label'] for entry in train_data], dtype=torch.float)

    val_input_ids = torch.tensor([entry['input_ids'] for entry in val_data], dtype=torch.long)
    val_attention_mask = torch.tensor([entry['attention_mask'] for entry in val_data], dtype=torch.long)
    val_token_type_ids = torch.tensor([entry['token_type_ids'] for entry in val_data], dtype=torch.long)
    val_labels = torch.tensor([entry['label'] for entry in val_data], dtype=torch.float)

    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_token_type_ids, val_labels)

    return train_dataset, val_dataset



if __name__ == "__main__":
    set_seed()
    train_dataset, val_dataset = load_data_from_pickle('roberta_train_data_binary.pkl', 'roberta_dev_data_binary.pkl')

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: LongAnswerTrainer(train_dataset, val_dataset, trial).train(), n_trials=30)

    print("Best Trial:", study.best_trial)
    print("Best F1 Score:", -study.best_value)
