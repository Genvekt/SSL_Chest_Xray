from torchvision import models
from torch import nn

def get_model(name, pretrained=False):
    model_class = getattr(models, name)
    model = model_class(pretrained=pretrained)
    model.name = name
    return model


def change_output(model, num_classes):
    if model.name == "resnet18":
        model.fc = nn.Linear(512, num_classes)
    elif model.name == "densenet121":
        model.classifier = nn.Linear(1024, num_classes)


def freeze_features(model):
    first_class_layer = 0
    if model.name == "resnet18":
        first_class_layer = 10

    elif model.name == "densenet121":
        first_class_layer = 2

    layer_num = 0
    for child in model.children():
        layer_num +=1
        if layer_num < first_class_layer:
            for param in child.parameters():
                param.requires_grad = False
                
