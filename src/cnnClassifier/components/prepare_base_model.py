import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from cnnClassifier.utils.common import save_model
#from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    #def __init__(self, config: PrepareBaseModelConfig):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load the pre-trained VGG16 model
    def get_base_model(self):
        self.model = models.vgg16(
            pretrained=self.config.params_weights,
            progress=True
        )

        # Move the model to the GPU if available
        self.model = self.model.to(self.device)
        save_model(path=self.config.base_model_path, model=self.model)

    
    # Prepare the full model by adding custom layers and freezing layers if necessary
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze all layers if required
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        # Freeze till specific layers
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in list(model.parameters())[:-freeze_till]:
                param.requires_grad = False

        # Add custom layers on top of VGG16
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, classes),
            nn.Softmax(dim=1)
        )

        # Move the model to the GPU
        #model = model.to(self.device)

        # Set up the optimizer and loss function (similar to compiling in Keras)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        return model, optimizer, criterion


    # Update the base model by applying modifications (freezing and adding new layers)
    def update_base_model(self):
        self.full_model, self.optimizer, self.criterion = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        save_model(path=self.config.updated_base_model_path, model=self.full_model)

        return self.full_model



