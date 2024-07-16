import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class ResNetClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = self.create_model()
        self.criterion = nn.BCEWithLogitsLoss()

    def create_model(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
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
            nn.Linear(128, 1)
        )
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=len(self.train_dataloader()), epochs=self.hparams.num_epochs)
        return [optimizer], [scheduler]

def prepare_data(args):
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

    train_data_dir = os.path.join(args.data_dir, "Training")
    test_data_dir = os.path.join(args.data_dir, "Test")

    full_train_dataset = datasets.ImageFolder(train_data_dir, transform=train_transforms)
    val_size = int(0.4 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    test_dataset = datasets.ImageFolder(test_data_dir, transform=valid_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_dataloader, val_dataloader, test_dataloader

def main(args):
    train_dataloader, val_dataloader, test_dataloader = prepare_data(args)
    model = ResNetClassifier(args)

    early_stopping = EarlyStopping(monitor='val_loss', patience=args.early_stopping_patience, mode='min')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.checkpoint_dir,
        filename='resnet50-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=args.num_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        log_every_n_steps=1,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model checkpoints")
    args = parser.parse_args()

    main(args)