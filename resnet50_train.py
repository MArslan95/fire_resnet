import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import json
from torch.cuda.amp import GradScaler, autocast
from evaluate_visualization import EvaluateVisualization
import torch.optim.lr_scheduler as lr_scheduler

class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.evaluator = EvaluateVisualization()
        self.model_name = self.model.__class__.__name__
        self.model_save_dir = f'./model_res_{self.model_name}'
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.best_metric_epoch = -1
        self.best_val_loss = float('inf')
        
        # Initialize result lists
        self.train_losses, self.val_losses, self.test_losses = [], [], []
        self.train_accuracies, self.val_accuracies, self.test_accuracies = [], [], []
        self.all_epoch_res = []

    def _train_one_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            
            if self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels.float().unsqueeze(1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float().unsqueeze(1))
                loss.backward()
                self.optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            predicted_train = torch.round(torch.sigmoid(outputs))
            correct_train += (predicted_train.cpu() == labels.cpu().unsqueeze(1)).sum().item()
            total_train += labels.size(0)

        accuracy_train = correct_train / total_train
        return epoch_train_loss / total_train, accuracy_train

    def _validate_one_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float().unsqueeze(1))
                total_loss += loss.item() * inputs.size(0)
                
                predicted = torch.round(torch.sigmoid(outputs))
                total_correct += (predicted.cpu() == labels.cpu().unsqueeze(1)).sum().item()
                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self._train_one_epoch()
            val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)
            test_loss, test_accuracy = self._validate_one_epoch(self.test_dataloader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.test_losses.append(test_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)
            self.test_accuracies.append(test_accuracy)

            self.scheduler.step()

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_metric_epoch = epoch + 1
                torch.save(self.model.state_dict(), f'{self.model_save_dir}/best_model_{self.model_name}.pth')

            torch.save(self.model.state_dict(), f'{self.model_save_dir}/latest_model_{self.model_name}.pth')

            epoch_res = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy,
                'test_acc': test_accuracy,
            }
            self.all_epoch_res.append(epoch_res)

        with open(self.results_file, 'w') as f:
            json.dump(self.all_epoch_res, f)

        self.evaluator.plot_loss_curve(self.train_losses, self.val_losses, self.model_name, self.model_save_dir)
        self.evaluator.plot_accuracy_curve(self.train_accuracies, self.val_accuracies, self.model_name, self.model_save_dir)
        self.evaluator.plot_test_acc_loss_curve(self.test_accuracies, self.test_losses, self.model_name, self.model_save_dir)

    def evaluate(self):
        self.model.load_state_dict(torch.load(f'{self.model_save_dir}/best_model_{self.model_name}.pth'))
        self.model.eval()
        y_true_test, y_pred_test = [], []
        correct_test, total_test, test_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                predicted = torch.round(torch.sigmoid(outputs))
                
                correct_test += (predicted.cpu() == labels.cpu().unsqueeze(1)).sum().item()
                total_test += labels.size(0)
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted.cpu().numpy())

                loss = self.criterion(outputs, labels.float().unsqueeze(1))
                test_loss += loss.item() * inputs.size(0)

        accuracy_test = correct_test / total_test
        test_loss /= total_test

        class_report = classification_report(y_true_test, y_pred_test, output_dict=True)
        class_report_df = pd.DataFrame(class_report).transpose()
        class_report_df.to_csv(f'{self.model_save_dir}/classification_report_{self.model_name}.csv', index=True)

        cm = confusion_matrix(y_true_test, y_pred_test)
        cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(len(cm))], columns=[f"Pred_{i}" for i in range(len(cm))])
        cm_df.to_csv(f'{self.model_save_dir}/confusion_matrix_{self.model_name}.csv')

        print(f"Confusion Matrix:\n{cm}")
        print(f"Classification Report:\n{class_report}")

        return test_loss, accuracy_test
s
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# import os
# import json
# from torch.cuda.amp import GradScaler, autocast
# from evaluate_visualization import EvaluateVisualization
# import torch.optim.lr_scheduler as lr_scheduler

# class ResNetTrainer:
#     def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda'):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.device = device
#         self.evaluator = EvaluateVisualization()
#         self.model_name = self.model.__class__.__name__
#         self.model_save_dir = f'./model_res_{self.model_name}'
#         os.makedirs(self.model_save_dir, exist_ok=True)
#         self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
#         self.scaler = GradScaler()
#         self.best_metric_epoch = -1
#         self.best_val_loss = float('inf')

#     def _train_one_epoch(self):
#         self.model.train()
#         epoch_train_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for inputs, labels in self.train_dataloader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)
#             self.optimizer.zero_grad()
            
#             with autocast():
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels.float().unsqueeze(1))
                
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             epoch_train_loss += loss.item() * inputs.size(0)
#             predicted_train = torch.round(torch.sigmoid(outputs))
#             correct_train += (predicted_train.cpu() == labels.cpu().unsqueeze(1)).sum().item()
#             total_train += labels.size(0)

#         accuracy_train = correct_train / total_train
#         return epoch_train_loss / total_train, accuracy_train
   
#     def _validate_one_epoch(self, dataloader):
#         self.model.eval()
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0

#         with torch.no_grad():
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels.float().unsqueeze(1))
#                 total_loss += loss.item() * inputs.size(0)
                
#                 predicted = torch.round(torch.sigmoid(outputs))
#                 total_correct += (predicted.cpu() == labels.cpu().unsqueeze(1)).sum().item()
#                 total_samples += labels.size(0)

#         average_loss = total_loss / total_samples
#         accuracy = total_correct / total_samples

#         return average_loss, accuracy

#     def train(self, num_epochs):
#         train_losses, val_losses, test_losses = [], [], []
#         train_accuracies, val_accuracies, test_accuracies = [], [], []
#         all_epoch_res = []

#         for epoch in range(num_epochs):
#             # Training
#             train_loss, train_accuracy = self._train_one_epoch()

#             # Validation
#             val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

#             # Test
#             test_loss, test_accuracy = self._validate_one_epoch(self.test_dataloader)

#             # Print progress
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

#             # Update the training, validation, and testing loss and accuracy lists
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             test_losses.append(test_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)
#             test_accuracies.append(test_accuracy)

#             # Scheduler step
#             self.scheduler.step()

#             # Save model and best epoch results
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self.best_metric_epoch = epoch + 1
#                 torch.save(self.model.state_dict(), f'{self.model_save_dir}/best_model_{self.model_name}.pth')

#             # Save the latest model
#             torch.save(self.model.state_dict(), f'{self.model_save_dir}/latest_model_{self.model_name}.pth')

#             epoch_res = {
#                 'epoch': epoch + 1,
#                 'train_loss': train_loss,
#                 'val_loss': val_loss,
#                 'test_loss': test_loss,
#                 'train_acc': train_accuracy,
#                 'val_acc': val_accuracy,
#                 'test_acc': test_accuracy,
#             }
#             all_epoch_res.append(epoch_res)

#         # Save all epoch results to a JSON file
#         with open(self.results_file, 'w') as f:
#             json.dump(all_epoch_res, f)

#         # Plot the training and validation loss and accuracy
#         self.evaluator.plot_loss_curve(train_losses, val_losses,self.model_name,self.model_save_dir)
#         self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies,self.model_name,self.model_save_dir)
#         self.evaluator.plot_test_acc_loss_curve(test_accuracies,test_losses,self.model_name,self.model_save_dir)

#     def evaluate(self):
#         # Load the best model
#         self.model.load_state_dict(torch.load(f'{self.model_save_dir}/best_model_{self.model_name}.pth'))
#         self.model.eval()
#         y_true_test, y_pred_test = [], []
#         correct_test, total_test, test_loss = 0, 0, 0.0

#         with torch.no_grad():
#             for inputs, labels in self.test_dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 predicted = torch.round(torch.sigmoid(outputs))
                
#                 correct_test += (predicted.cpu() == labels.cpu().unsqueeze(1)).sum().item()
#                 total_test += labels.size(0)
#                 y_true_test.extend(labels.cpu().numpy())
#                 y_pred_test.extend(predicted.cpu().numpy())

#                 # Calculate test loss
#                 loss = self.criterion(outputs, labels.float().unsqueeze(1))
#                 test_loss += loss.item() * inputs.size(0)

#         accuracy_test = correct_test / total_test
#         test_loss /= total_test  # Calculate average test loss

#         # Generate classification report
#         class_report = classification_report(y_true_test, y_pred_test, output_dict=True)
#         class_report_df = pd.DataFrame(class_report).transpose()
#         class_report_df.to_csv(f'{self.model_save_dir}/classification_report_{self.model_name}.csv', index=True)

#         # Save confusion matrix
#         cm = confusion_matrix(y_true_test, y_pred_test)
#         cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(len(cm))], columns=[f"Pred_{i}" for i in range(len(cm))])
#         cm_df.to_csv(f'{self.model_save_dir}/confusion_matrix_{self.model_name}.csv')

#         print(f"Confusion Matrix:\n{cm}")
#         print(f"Classification Report:\n{class_report}")

#         return test_loss, accuracy_test


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms, datasets
# from evaluate_visualization import EvaluateVisualization
# from torch.cuda.amp import GradScaler, autocast
# from sklearn.metrics import classification_report, confusion_matrix
# import torch.optim.lr_scheduler as lr_scheduler
# import pandas as pd
# import os
# import json


# class ResNetTrainer:
#     def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, scheduler, device='cuda'):
#         self.model = model.to(device)
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.test_dataloader = test_dataloader
#         self.criterion = criterion
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.device = device
#         self.evaluator = EvaluateVisualization()
#         self.model_name = self.model.__class__.__name__
#         self.model_save_dir = f'./model_res_{self.model_name}'
#         os.makedirs(self.model_save_dir, exist_ok=True)
#         self.results_file = f'{self.model_save_dir}/All_epochs_results_{self.model_name}.json'
#         self.scaler = GradScaler()
#         self.best_metric_epoch = -1
#         self.best_val_loss = float('inf')

#     def _train_one_epoch(self):
#         self.model.train()
#         epoch_train_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         for inputs, labels in self.train_dataloader:
#             inputs, labels = inputs.to(self.device), labels.to(self.device)
#             self.optimizer.zero_grad()
            
#             with autocast():
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
                
#             self.scaler.scale(loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             epoch_train_loss += loss.item() * inputs.size(0)
#             _, predicted_train = torch.max(outputs.data, 1)
#             correct_train += (predicted_train == labels).sum().item()
#             total_train += labels.size(0)

#         accuracy_train = correct_train / total_train
#         return epoch_train_loss / total_train, accuracy_train
   
#     def _validate_one_epoch(self, dataloader):
#         self.model.eval()
#         total_loss = 0.0
#         total_correct = 0
#         total_samples = 0

#         with torch.no_grad():
#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 total_loss += loss.item() * inputs.size(0)
                
#                 _, predicted = torch.max(outputs.data, 1)
#                 total_correct += (predicted == labels).sum().item()
#                 total_samples += labels.size(0)

#         average_loss = total_loss / total_samples
#         accuracy = total_correct / total_samples

#         return average_loss, accuracy

#     def train(self, num_epochs):
#         train_losses, val_losses, test_losses = [], [], []
#         train_accuracies, val_accuracies, test_accuracies = [], [], []
#         all_epoch_res = []

#         for epoch in range(num_epochs):
#             # Training
#             train_loss, train_accuracy = self._train_one_epoch()

#             # Validation
#             val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

#             # Test
#             test_loss, test_accuracy = self._validate_one_epoch(self.test_dataloader)

#             # Print progress
#             print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

#             # Update the training, validation, and testing loss and accuracy lists
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             test_losses.append(test_loss)
#             train_accuracies.append(train_accuracy)
#             val_accuracies.append(val_accuracy)
#             test_accuracies.append(test_accuracy)

#             epoch_data = {
#                 'epoch': epoch + 1,
#                 'training_loss': train_loss,
#                 'training_accuracy': train_accuracy,
#                 'validation_loss': val_loss,
#                 'validation_accuracy': val_accuracy,
#                 'testing_loss': test_loss,
#                 'testing_accuracy': test_accuracy,
#             }
#             all_epoch_res.append(epoch_data)

#             # Save best model based on validation loss
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 self.best_metric_epoch = epoch
#                 torch.save(self.model.state_dict(), f"{self.model_save_dir}/{self.model_name}_best.pth")

#             # Step with the scheduler based on validation loss
#             if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
#                 self.scheduler.step(val_loss)
#             else:
#                 self.scheduler.step()

    
#         # Save all epoch results to a JSON file
#         with open(self.results_file, 'w') as f:
#             json.dump(all_epoch_res, f, indent=4)

#         # Plotting loss and accuracy curves
#         self.evaluator.plot_loss_curve(train_losses, val_losses)
#         self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)
#         self.evaluator.plot_test_acc_loss_curve(test_losses, test_accuracies)

#     def evaluate(self):
#         self.model.load_state_dict(torch.load(f"{self.model_save_dir}/{self.model_name}_best.pth"))
#         self.model.eval()
#         y_true_test, y_pred_test = [], []
#         correct_test, total_test, test_loss = 0, 0, 0.0

#         with torch.no_grad():
#             for inputs, labels in self.test_dataloader:
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#                 outputs = self.model(inputs)

#                 _, predicted_test = torch.max(outputs.data, 1)
#                 correct_test += (predicted_test == labels).sum().item()
#                 total_test += labels.size(0)
#                 y_true_test.extend(labels.cpu().numpy())
#                 y_pred_test.extend(predicted_test.cpu().numpy())

#                 # Calculate test loss
#                 loss = self.criterion(outputs, labels)
#                 test_loss += loss.item() * inputs.size(0)

#         accuracy_test = correct_test / total_test
#         test_loss /= total_test  # Calculate average test loss

#         # Generate classification report
#         class_report = classification_report(y_true_test, y_pred_test, output_dict=True)
#         class_report_df = pd.DataFrame(class_report).transpose()
#         class_report_df.to_csv(f'{self.model_save_dir}/classification_report_epoch_{self.best_metric_epoch}.csv', index=True)

#         # Save confusion matrix
#         cm = confusion_matrix(y_true_test, y_pred_test)
#         cm_df = pd.DataFrame(cm, index=[f"True_{i}" for i in range(len(cm))], columns=[f"Pred_{i}" for i in range(len(cm))])
#         cm_df.to_csv(f'{self.model_save_dir}/confusion_matrix_epoch_{self.best_metric_epoch}.csv')

#         return test_loss, accuracy_test
    