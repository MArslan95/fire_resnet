import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from evaluate_visualization import EvaluateVisualization  # Assuming this is a custom module
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import classification_report, confusion_matrix
import torch.optim.lr_scheduler as lr_scheduler
import pandas as pd
import os
import json
from typing import Tuple, Any

class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataloader: DataLoader, 
                 val_dataloader: DataLoader, 
                 test_dataloader: DataLoader, 
                 criterion: nn.Module, 
                 optimizer: optim.Optimizer, 
                 scheduler: Any, 
                 device: str = 'cuda', 
                 validate_freq: int = 1, 
                 save_intermediate: bool = True, 
                 early_stopping_patience: int = 10):
        """
        Initialize the ResNetTrainer with model, dataloaders, criterion, optimizer, scheduler, and device.
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.validate_freq = validate_freq
        self.save_intermediate = save_intermediate
        self.early_stopping_patience = early_stopping_patience

        self.evaluator = EvaluateVisualization()
        self.model_name = self.model.__class__.__name__
        self.model_save_dir = f'./model_res_{self.model_name}'
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
        self.scaler = GradScaler()
        self.best_metric_epoch = -1
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        # Log the initialization parameters
        self._log_params()

    def _log_params(self):
        params = {
            'model_name': self.model_name,
            'optimizer': str(self.optimizer),
            'criterion': str(self.criterion),
            'scheduler': str(self.scheduler),
            'device': self.device,
            'validate_freq': self.validate_freq,
            'save_intermediate': self.save_intermediate,
            'early_stopping_patience': self.early_stopping_patience
        }
        with open(f'{self.model_save_dir}/training_params.json', 'w') as f:
            json.dump(params, f, indent=4)

    def _train_one_epoch(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        """
        self.model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs.data, 1)
            correct_train += (predicted_train == labels).sum().item()
            total_train += labels.size(0)

        accuracy_train = correct_train / total_train
        return epoch_train_loss / total_train, accuracy_train

    def _validate_one_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model for one epoch on a given dataloader.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy

    def train(self, num_epochs: int):
        """
        Train the model for a specified number of epochs.
        """
        train_losses, val_losses, test_losses = [], [], []
        train_accuracies, val_accuracies, test_accuracies = [], [], []
        all_epoch_res = []

        for epoch in range(num_epochs):
            # Training
            train_loss, train_accuracy = self._train_one_epoch()

            if epoch % self.validate_freq == 0:
                # Validation
                val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

                # Test
                test_loss, test_accuracy = self._validate_one_epoch(self.test_dataloader)

                # Print progress
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

                # Update the training, validation, and testing loss and accuracy lists
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_accuracy)
                val_accuracies.append(val_accuracy)
                test_accuracies.append(test_accuracy)

                epoch_data = {
                    'epoch': epoch + 1,
                    'training_loss': train_loss,
                    'training_accuracy': train_accuracy,
                    'validation_loss': val_loss,
                    'validation_accuracy': val_accuracy,
                    'testing_loss': test_loss,
                    'testing_accuracy': test_accuracy,
                }
                all_epoch_res.append(epoch_data)

                # Save best model based on validation loss
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_metric_epoch = epoch
                    torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model_name}_best.pth")
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1

                # Step with the scheduler based on validation loss
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

                # Check for early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping")
                    break

                # Save intermediate models if specified
                if self.save_intermediate and (epoch + 1) % 10 == 0:
                    torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model_name}_epoch_{epoch + 1}.pth")

        # Save all epoch results to a JSON file
        with open(self.results_file, 'w') as f:
            json.dump(all_epoch_res, f, indent=4)

        # Plotting loss and accuracy curves
        self.evaluator.plot_loss_curve(train_losses, val_losses)
        self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)
        self.evaluator.plot_test_acc_loss_curve(test_losses, test_accuracies)

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on the test dataset and generate a classification report.
        """
        self.model.load_state_dict(torch.load(f"{self.model_save_dir}/{self.model_name}_best.pth"))
        self.model.eval()
        y_true_test, y_pred_test = [], []
        correct_test, total_test, test_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                _, predicted_test = torch.max(outputs.data, 1)
                correct_test += (predicted_test == labels).sum().item()
                total_test += labels.size(0)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted_test.cpu().numpy())

                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

        accuracy_test = correct_test / total_test
        average_test_loss = test_loss / total_test

        print("Test Accuracy: ", accuracy_test)
        print("Test Loss: ", average_test_loss)
        print("Classification Report: \n", classification_report(y_true_test, y_pred_test))
        print("Confusion Matrix: \n", confusion_matrix(y_true_test, y_pred_test))

        return average_test_loss, accuracy_test
