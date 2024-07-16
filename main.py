import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim.lr_scheduler as lr_scheduler
from resnet50_train import ResNetTrainer  # Assuming ResNetTrainer class is defined in 'resnet34_train.py'

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, valid_transforms

def get_datasets(train_data_dir, test_data_dir, train_transforms, valid_transforms):
    full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
    val_size = int(0.3 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)
    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=12):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader

def create_model():
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
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
        nn.Linear(128, 1)  # Single output neuron for binary classification
    )

    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def main():
    train_transforms, valid_transforms = get_transforms()

    train_data_dir = "./dataset/Training"
    test_data_dir = "./dataset/Test"

    train_dataset, val_dataset, test_dataset = get_datasets(train_data_dir, test_data_dir, train_transforms, valid_transforms)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dataset, val_dataset, test_dataset)

    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Number of validation samples: {len(val_dataloader.dataset)}")
    print(f"Number of testing samples: {len(test_dataloader.dataset)}")

    model = create_model()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dataloader), epochs=100)

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
# from torchvision.models import resnet50, ResNet50_Weights
# import torch.optim.lr_scheduler as lr_scheduler
# from resnet50_train import ResNetTrainer  # Assuming ResNetTrainer class is defined in 'resnet34_train.py'


# def main():
#     # Data augmentation transformations
#     train_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(45),
#         # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Dataset directories
#     train_data_dir = "./dataset/Training"
#     test_data_dir = "./dataset/Test"
    
#     # train_data_dir = "./forest_fire_dataset/Training"
#     # valid_data_dir = "./forest_fire_dataset/Test"
#     # test_data_dir = "./forest_fire_dataset/Test"


# #     # Create datasets and dataloaders
#     # train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     # val_dataset = datasets.ImageFolder(valid_data_dir, transform=valid_transforms)
    

#     # Create datasets
#     full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     val_size = int(0.3 * len(full_train_dataset))
#     train_size = len(full_train_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)

#     # Create dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
#     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)
#     test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)

#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # Create ResNet50 model
#     model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     for param in model.parameters():
#         param.requires_grad = False
        
#     num_features = model.fc.in_features

#     # Modify the classifier (fully connected layers)
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
#         nn.Linear(128, 1)# Single output neuron for binary classification
#         # nn.Softmax(dim=1)
#     )

#     # Enable gradient calculation for the classifier layers only
#     for param in model.fc.parameters():
#         param.requires_grad = True

#     # Define loss function and optimizer
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

#     # Create scheduler
#     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dataloader), epochs=100)

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)
#     model.to(device)

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

#     num_epochs = 5
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()



























# def main():
#     # Data augmentation transformations
#     train_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomRotation(45),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     valid_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     # Dataset directories
#     train_data_dir = "./dataset/Training"
#     test_data_dir = "./dataset/Test"

#     # Create datasets
#     full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
#     val_size = int(0.4 * len(full_train_dataset))
#     train_size = len(full_train_dataset) - val_size
#     train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
#     test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)

#     # Create dataloaders
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)
#     test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=12)

#     print(f"Number of training samples: {len(train_dataloader.dataset)}")
#     print(f"Number of validation samples: {len(val_dataloader.dataset)}")
#     print(f"Number of testing samples: {len(test_dataloader.dataset)}")

#     # # Create ResNet34 model
#     model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#     num_features = model.fc.in_features
    
#     # # Modify the classifier (fully connected layers)
#     # model.fc = nn.Sequential(
#     #     nn.Linear(num_features, 1024),
#     #     nn.BatchNorm1d(1024),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.4),
#     #     nn.Linear(1024, 512),
#     #     nn.BatchNorm1d(512),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.3),
#     #     nn.Linear(512, 256),
#     #     nn.BatchNorm1d(256),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.2),
#     #     nn.Linear(256, 128),
#     #     nn.BatchNorm1d(128),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.1),
#     #     nn.Linear(128, 2)
#     # )

#     # # Enable gradient calculation for the classifier layers only
#     for param in model.fc.parameters():
#         param.requires_grad = True
#     # model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#     # num_features = model.classifier[1].in_features
    
#     # Modify the classifier (fully connected layers)
#     # model.classifier = nn.Sequential(
#     #     nn.Linear(num_features, 1024),
#     #     nn.BatchNorm1d(1024),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.4),
#     #     nn.Linear(1024, 512),
#     #     nn.BatchNorm1d(512),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.3),
#     #     nn.Linear(512, 256),
#     #     nn.BatchNorm1d(256),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.2),
#     #     nn.Linear(256, 128),
#     #     nn.BatchNorm1d(128),
#     #     nn.ReLU(inplace=True),
#     #     nn.Dropout(p=0.1),
#     #     nn.Linear(128, 2),
#     #     nn.Softmax(dim=1)
        
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
#         nn.Linear(128, 1)  # Single output neuron for binary classification
#     )

#     # Enable gradient calculation for the classifier layers only
#     for param in model.fc.parameters():
#         param.requires_grad = True

#     # Define loss function and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)  # Slightly increased weight decay

#     # Create scheduler
#     scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_dataloader), epochs=100)

#     # Training and evaluation
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(device)
#     model.to(device)

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

#     num_epochs = 100
#     trainer.train(num_epochs)
#     trainer.evaluate()

# if __name__ == "__main__":
#     main()
