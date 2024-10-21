import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import json
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.utils.common import read_yaml, create_directories,save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    # Function to load validation data
    def _valid_generator(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        valid_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        self.valid_generator = DataLoader(
            valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    # Static method to load the saved PyTorch model
    @staticmethod
    def load_model(path: Path) -> nn.Module:
        # model = models.vgg16(pretrained=False)  # Load the same model architecture
        # model.classifier[6] = nn.Linear(4096, 2) 
        x = ConfigurationManager()
        conf = x.get_prepare_base_model_config()
        obj_model = PrepareBaseModel(conf)
        obj_model.get_base_model()
        model = obj_model.update_base_model()
        model.load_state_dict(torch.load(path))
        model.eval()  # Set model to evaluation mode
        return model

    # Method to evaluate the model on validation data
    def evaluation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.config.path_of_model).to(device)
        self._valid_generator()
        self.score = self._evaluate_model(device)
        self.save_score()

    # Evaluate model performance on validation data
    def _evaluate_model(self, device):
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in self.valid_generator:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    # Save evaluation scores to a JSON file
    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)   

    # Log metrics and model to MLflow
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # Check if the model registry can work with the current URI
            if tracking_url_type_store != "file":
                # Register the model to MLflow registry
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                # Log model without registry
                mlflow.pytorch.log_model(self.model, "model")