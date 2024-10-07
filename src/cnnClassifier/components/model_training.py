import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
from cnnClassifier.utils.common import save_model
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def base_model(self):
        x = ConfigurationManager()
        conf = x.get_prepare_base_model_config()
        obj_model = PrepareBaseModel(conf)
        obj_model.get_base_model()
        self.model = obj_model.update_base_model()
        self.model.load_state_dict(torch.load(self.config.updated_base_model_path))

        # Move the model to the GPU if available
        self.model = self.model.to(self.device)


    def train_valid_generator(self):
        # Data transformations (with or without augmentation)
        if self.config.params_is_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.config.params_image_size[0]),
                transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Datasets
        train_dataset = datasets.ImageFolder(self.config.training_data, transform=train_transform)
        valid_dataset = datasets.ImageFolder(self.config.training_data, transform=valid_transform)
        # validation_split=0.20

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    
    @staticmethod
    def save_model(self, path: Path):
        torch.save(self.model.state_dict(), path)

    
    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(self.config.params_epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.config.params_epochs}], Loss: {running_loss / len(self.train_loader)}")

            # Validation
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, labels in self.valid_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Validation Accuracy: {100 * correct / total}%')

        # Save the trained model
        save_model(path=self.config.trained_model_path, model=self.model)

    