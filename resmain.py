import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet34, ResNet34_Weights
import torch.optim.lr_scheduler as lr_scheduler
from resnet50_train import ResNetTrainer  # Assuming ResNetTrainer class is defined in 'resnet34_train.py'

def main():
    # Data augmentation transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),  # Increased rotation angle
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Add color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset directories
    train_data_dir = "./dataset/Training"
    test_data_dir = "./dataset/Test"

    # Create datasets
    full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
    val_size = int(0.4 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Number of validation samples: {len(val_dataloader.dataset)}")
    print(f"Number of testing samples: {len(test_dataloader.dataset)}")

    # Create ResNet34 model
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    
    # Modify the classifier (fully connected layers)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(128, 2)
    )

    # Enable gradient calculation for the classifier layers only
    for param in model.fc.parameters():
        param.requires_grad = True

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001, weight_decay=0.0001)

    # Create scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

    # Training and evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model.to(device)

    trainer = ResNetTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    num_epochs = 100
    trainer.train(num_epochs)
    trainer.evaluate()

if __name__ == "__main__":
    main()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets
# from torchvision.models import resnet34, ResNet34_Weights
# import torch.optim.lr_scheduler as lr_scheduler
# from resnet50_train import ResNetTrainer

# def main():
#     # Data augmentation transformations
#     # train_transforms = transforms.Compose([
#     #     transforms.RandomResizedCrop(224),
#     #     transforms.RandomHorizontalFlip(),
#     #     transforms.RandomRotation(20),  # Increased rotation
#     #     transforms.ToTensor(),
#     #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     # ])
#     train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(30),  # Increased rotation angle
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Add color jitter
#     # transforms.RandomGrayscale(p=0.1),  # Randomly convert images to grayscale
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Dataset directories
#     train_data_dir = "./dataset/Training"
#     # val_data_dir = "./dataset/Test"
#     test_data_dir = "./dataset/Test"

#     # Create datasets
#     full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     # train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     # val_dataset = datasets.ImageFolder(val_data_dir, transform=valid_transforms)
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)

#     # Split the training dataset into training and validation datasets
#     val_size = int(0.4 * len(full_train_dataset))
#     train_size = len(full_train_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

#     # Create dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
#     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # Create ResNet50 model
#     model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    
#     for param in model.parameters():
#         param.requires_grad = False

#     num_features = model.fc.in_features
#     # Modify the classifier
#     # model.fc = nn.Sequential(
#     #     nn.Linear(num_features, 1024),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.4),
#     #     nn.Linear(1024, 512),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.3),
#     #     nn.Linear(512, 128),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.2),
#     #     nn.Linear(128, 2),
#     #     nn.LogSoftmax(dim=1)
#     # )
#     model.fc = nn.Sequential(
#         nn.Linear(num_features, 1024),
#         nn.BatchNorm1d(1024),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.4),
#         nn.Linear(1024, 512),
#         nn.BatchNorm1d(512),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.3),
#         nn.Linear(512, 256),
#         nn.BatchNorm1d(256),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.2),
#         nn.Linear(256, 128),
#         nn.BatchNorm1d(128),
#         nn.ReLU(inplace=True),
#         nn.Dropout(p=0.1),
#         nn.Linear(128, 2)
#     )



#     for param in model.fc.parameters():
#         param.requires_grad = True

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.fc.parameters(), lr=0.0001, weight_decay=0.0001)
   
#     # Create dynamic scheduler
#     # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
#     scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     print(device)

#     trainer = ResNetTrainer(
#         model=model,
#         train_dataloader=train_dataloader,
#         val_dataloader=val_dataloader,
#         test_dataloader=test_dataloader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#     )

#     num_epochs = 50
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet18,ResNet18_Weights ,resnet34,ResNet34_Weights, resnet50,ResNet50_Weights
import torch.optim.lr_scheduler as lr_scheduler
from resnet50_train import ResNetTrainer


def main():
    # Data augmentation transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # Increased rotation
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Increased jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # # Dataset directories
    
    train_data_dir = "./dataset/Training"
    valid_data_dir = "./dataset/Training"
    test_data_dir = "./dataset/Test"


    # Create datasets and dataloaders
    train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms, validation=0.2)
    val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms, validation=0.2)
    test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Number of validation samples: {len(val_dataloader.dataset)}")
    print(f"Number of testing samples: {len(test_dataloader.dataset)}")

    # Create ResNet50 model
    # model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
    # model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    # Modify the classifier
    model.fc = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.6),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(4096, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(2048, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 128),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(128, 2),
        nn.Softmax(dim=1)
    )
   
    for param in model.fc.parameters():
        param.requires_grad = True
        
    # num_features = model.classifier[0].in_features
    # # Modify the classifier
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.6),
        nn.Linear(4096, 128),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(128,2),
        # nn.Sigmoid(),
        nn.Softmax(dim=1)
    )
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    
    
    # Ensure only the classifier parameters are trainable
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=0.01,eps=1e-08, weight_decay=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
   
    # Create dynamic scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=1e-4)

    # Training and evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    trainer = ResNetTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    num_epochs = 100
    trainer.train(num_epochs)
    trainer.evaluate()

if __name__ == "__main__":
    main()
    