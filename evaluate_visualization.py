import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os

class EvaluateVisualization:
    @staticmethod
    def plot_loss_curve(train_losses, val_losses, model_name, model_save_dir):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
        plt.plot(epochs, val_losses, 'go-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        file_name = f"{model_name}_epochs_{len(epochs)}_loss.png"
        plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

    @staticmethod
    def plot_accuracy_curve(train_accuracies, val_accuracies, model_name, model_save_dir):
        epochs = range(1, len(train_accuracies) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
        plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        file_name = f"{model_name}_epochs_{len(epochs)}_acc.png"
        plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

    @staticmethod
    def plot_test_acc_loss_curve(test_accuracies, test_losses, model_name, model_save_dir):
        epochs = range(1, len(test_accuracies) + 1)
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Plot Test Accuracy
        ax1.plot(epochs, test_accuracies, 'bo-', label='Test Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis for Test Loss
        ax2 = ax1.twinx()
        ax2.plot(epochs, test_losses, 'go-', label='Test Loss')
        ax2.set_ylabel('Loss', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # Title and Grid
        plt.title('Test Accuracy and Loss')
        ax1.grid(True)

        # Combine Legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')

        plt.tight_layout()
        file_name = f"{model_name}_epochs_{len(epochs)}_test_acc_loss.png"
        plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, model_name, model_save_dir):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        file_name = f"{model_name}_confusion_matrix.png"
        plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

    @staticmethod
    def plot_classification_report(y_true, y_pred, classes, model_name, model_save_dir):
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        plt.figure(figsize=(10, 7))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues')
        plt.title('Classification Report')
        file_name = f"{model_name}_classification_report.png"
        plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)




# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import classification_report, confusion_matrix
# import pandas as pd
# import os

# class EvaluateVisualization:
#     @staticmethod
#     def plot_loss_curve(train_losses, val_losses,model_name,model_save_dir):
#         epochs = range(1, len(train_losses) + 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
#         plt.plot(epochs, val_losses, 'go-', label='Validation Loss')
#         plt.title('Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.ylim(0, 1)
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()  # Adjust layout for better spacing
#         # plt.show()
#         file_name = f"{model_name}_epoches_{epochs}_loss.png"
#         plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

#     @staticmethod  
#     def plot_accuracy_curve(train_accuracies, val_accuracies,model_name,model_save_dir):
#         epochs = range(1, len(train_accuracies) + 1)
        
#         plt.figure(figsize=(10, 6))
#         plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
#         plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
        
#         plt.title('Training and Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.ylim(0, 1)  # Assuming accuracy ranges from 0 to 1
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()  # Adjust layout for better spacing
#         # plt.show()
#         file_name = f"{model_name}_epoches_{epochs}_acc.png"
#         plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
    
#     @staticmethod  
#     def plot_test_acc_loss_curve(test_accuracies, test_losses,model_name,model_save_dir):
#         epochs = range(1, len(test_accuracies) + 1)
        
#         fig, ax1 = plt.subplots(figsize=(12, 7))

#         # Plot Test Accuracy
#         ax1.plot(epochs, test_accuracies, 'bo-', label='Test Accuracy')
#         ax1.set_xlabel('Epochs')
#         ax1.set_ylabel('Accuracy', color='b')
#         ax1.set_ylim(0, 1)  # Assuming accuracy ranges from 0 to 1
#         ax1.tick_params(axis='y', labelcolor='b')

#         # Create a second y-axis for Test Loss
#         ax2 = ax1.twinx()
#         ax2.plot(epochs, test_losses, 'go-', label='Test Loss')
#         ax2.set_ylabel('Loss', color='g')
#         ax2.tick_params(axis='y', labelcolor='g')
        
#         # Title and Grid
#         plt.title('Test Accuracy and Loss')
#         ax1.grid(True)

#         # Combine Legends
#         lines, labels = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines + lines2, labels + labels2, loc='best')

#         plt.tight_layout()  # Adjust layout for better spacing
#         # plt.show()
#         file_name = f"{model_name}_epoches_{epochs}_test_acc_loss.png"
#         plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

#     @staticmethod
#     def plot_confusion_matrix(y_true, y_pred, classes,model_name,model_save_dir):
#         cm = confusion_matrix(y_true, y_pred)
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         # plt.show()
#         file_name = f"{model_name}_confusion_matrix.png"
#         plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)

#     @staticmethod
#     def plot_classification_report(y_true, y_pred, classes,model_name,model_save_dir):
#         report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
#         report_df = pd.DataFrame(report).transpose()
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap='Blues')
#         plt.title('Classification Report')
#         # plt.show()
#         file_name = f"{model_name}_classification_report.csv"
#         plt.savefig(os.path.join(model_save_dir, file_name), dpi=500, format='png', bbox_inches='tight', pad_inches=0)
