import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertForSequenceClassification, AdamW
from torch.utils.tensorboard import SummaryWriter
import math
from yes_no_dataset import YesNoDataset  # Assuming YesNoDataset is in yes_no_dataset.py


class YesNoBertTrainer:
    def __init__(self, train_file, val_file, batch_size=16, lr=2e-5, num_epochs=3, device=None):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir='runs/yes_no_bert_training')

        # Load the datasets
        train_dataset = YesNoDataset(train_file)
        val_dataset = YesNoDataset(val_file)

        # Initialize BERT model for binary classification
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

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
        n_epochs_stop = 3  # Early stopping after 3 epochs of no improvement

        print("Started training")
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0

            for batch in self.train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.train_loader)
            train_accuracy = self.evaluate_accuracy(self.train_loader)

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

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()

                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

        avg_loss = epoch_loss / len(self.val_loader)
        accuracy = correct_predictions / total_predictions
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
    # File paths for training and validation datasets
    train_file = 'datasets/simplified-yes-no-train.jsonl'
    val_file = 'datasets/simplified-yes-no-dev.jsonl'

    # Initialize trainer with training and validation datasets
    trainer = YesNoBertTrainer(train_file=train_file, val_file=val_file, batch_size=16, lr=2e-5, num_epochs=5)
    trainer.train()
