import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from yes_no_dataset import YesNoDataset  # Assuming YesNoDataset is in yes_no_dataset.py
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import json

class YesNoBertTrainer:
    def __init__(self, data_file, batch_size=8, lr=1e-5, num_epochs=10, device=None, max_length=128):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir='runs/yes_no_bert_training')

        # Load data from the file
        with open(data_file, 'r') as f:
            raw_data = [json.loads(line.strip()) for line in f if line.strip()]

        # Split data into training and validation sets
        train_data, val_data = train_test_split(raw_data, test_size=0.15, random_state=42)

        print(f"Total data entries: {len(raw_data)}")
        print(f"Training data entries: {len(train_data)}")
        print(f"Validation data entries: {len(val_data)}")

        # Create datasets
        train_dataset = YesNoDataset(train_data, max_length=max_length, balance=True)
        val_dataset = YesNoDataset(val_data, max_length=max_length, balance=False)

        # Get labels from the training data (after oversampling)
        train_labels = [label.item() for label in train_dataset.labels]

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        # Initialize BERT model for binary classification
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            hidden_dropout_prob=0.5  # Increased dropout
        ).to(self.device)

        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True


        # Optimizer and scheduler
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=len(train_dataset) // self.batch_size * self.num_epochs
        )

        # Gradient scaler for mixed precision
        self.scaler = GradScaler()

        # Dataloaders for training and validation sets
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

    def train(self):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        n_epochs_stop = 5

        print("Started training")
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

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                epoch_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

            avg_loss = epoch_loss / len(self.train_loader)
            train_accuracy = correct_predictions / total_predictions

            val_loss, val_accuracy = self.evaluate()

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], "
                  f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/Train', avg_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)

            # Early stopping and model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), 'best_validation_loss_model.pth')
                print("Model saved as best_validation_loss_model.pth")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= n_epochs_stop:
                    print(f"Early stopping triggered after {n_epochs_stop} epochs with no improvement.")
                    break

        self.writer.close()

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

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions

        # Compute additional metrics
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        print(f"Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

        return avg_loss, accuracy

    def evaluate_accuracy(self, loader):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        accuracy = correct_predictions / total_predictions
        return accuracy


if __name__ == "__main__":
    # File path for the data
    data_file = 'simplified-yes-no-train.jsonl'

    # Initialize trainer with data
    trainer = YesNoBertTrainer(
        data_file=data_file,
        batch_size=16,  # Increased batch size
        lr=1e-5,        # Adjusted learning rate if needed
        num_epochs=10,  # Increased number of epochs
        max_length=256
    )
    trainer.train()
