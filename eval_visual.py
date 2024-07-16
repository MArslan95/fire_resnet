
# import torch
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Evaluation and Visualization class
# class EvaluateVisualization:
#     @staticmethod
#     def plot_loss_curve(train_losses, val_losses):
#         epochs = range(1, len(train_losses) + 1)
#         plt.figure(figsize=(10, 5))
#         plt.plot(epochs,train_losses,'bo-',label='Training Loss',)
#         plt.plot(epochs,val_losses,'go-', label='Validation Loss')
#         plt.title('Training and Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.ylim(0, 1) 
#         # plt.xticks(range(1, len(train_losses) + 1))  # Show all epochs on x-axis
#         plt.grid(True)  # Show grid for better readability
#         plt.legend()
#         plt.show()
    
#     @staticmethod  
#     def plot_accuracy_curve(train_accuracies, val_accuracies):
#         epochs = range(1, len(train_accuracies) + 1)
        
#         plt.figure(figsize=(10, 6))  # Adjusted figure size for better aspect ratio
#         plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
#         plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
        
#         plt.title('Training and Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.ylim(0, 1)  # Assuming accuracy ranges from 0 to 1, adjust if necessary
        
#         # plt.xticks(range(1, len(train_accuracies) + 1))  # Show all epochs on x-axis
#         plt.grid(True)  # Show grid for better readability
        
#         plt.legend()
#         plt.tight_layout()  # Adjust layout for better spacing
#         plt.show()   
   
#     @staticmethod  
#     def plot_test_acc_loss_curve(test_accuracies, test_losses):
#         epochs = range(1, len(test_accuracies) + 1)
        
#         fig, ax1 = plt.subplots(figsize=(12, 7))  # Adjusted figure size for better aspect ratio

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
#         plt.title('Training and Validation Accuracy')
#         ax1.grid(True)  # Show grid for better readability

#         # Combine Legends
#         lines, labels = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines + lines2, labels + labels2, loc='best')

#         plt.tight_layout()  # Adjust layout for better spacing
#         plt.show()
#     # @staticmethod  
#     # def  plot_test_acc_loss_curve(test_accuracies, test_losses):
#     #     epochs = range(1, len(test_accuracies) + 1)
        
#     #     plt.figure(figsize=(10, 6))  # Adjusted figure size for better aspect ratio
#     #     plt.plot(epochs, test_accuracies, 'bo-', label='Test Accuracy')
#     #     plt.plot(epochs, test_losses, 'go-', label='Test Loss')
        
#     #     plt.title('Training and Validation Accuracy')
#     #     plt.xlabel('Epochs')
#     #     plt.ylabel('Accuracy')
#     #     plt.ylim(0, 1)  # Assuming accuracy ranges from 0 to 1, adjust if necessary
        
#     #     plt.xticks(range(1, len(test_accuracies) + 1))  # Show all epochs on x-axis
#     #     plt.grid(True)  # Show grid for better readability
        
#     #     plt.legend()
#     #     plt.tight_layout()  # Adjust layout for better spacing
#     #     plt.show()
        
    
