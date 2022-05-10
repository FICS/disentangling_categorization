import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import community.jalal_models as jalal_models


class Base_override(nn.Module):
    """
    Base class for overriding ConceptWhitening models for feature output. 
    """
    def __init__(self, base_model):
        super(Base_override, self).__init__()
        self.base_model = base_model
    
    def forward_feats(self, x):
        # logits, (None feats, None structure)
        return self.base_model(x), (None, None)
    
    def forward(self, x):
        logits, x = self.forward_feats(x)
        return logits
    

class ResNet_features(Base_override):
    def __init__(self, base_model):
        super(ResNet_features, self).__init__(base_model)
        self.classifier = self.base_model.fc
        
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    """
    def forward_feats(self, x):
        base = self.base_model
        
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        x = base.maxpool(x)

        x = base.layer1(x)
        x = base.layer2(x)
        x = base.layer3(x)
        x = base.layer4(x)

        x = base.avgpool(x)
        x = torch.flatten(x, 1)
        logits = base.fc(x)

        return logits, (x, torch.zeros(x.shape[0], 1))
    
    
class DenseNet_features(Base_override):
    def __init__(self, base_model):
        super(DenseNet_features, self).__init__(base_model)
        self.classifier = self.base_model.classifier
        
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html
    """
    def forward_feats(self, x):
        base = self.base_model
        
        features = base.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = base.classifier(out)
        
        return logits, (out, torch.zeros(out.shape[0], 1))

    
class VGG_features(Base_override):
    def __init__(self, base_model):
        super(VGG_features, self).__init__(base_model)
        self.classifier = self.base_model.classifier[6]
        
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
    """
    def forward_feats(self, x):
        base = self.base_model
        
        x = base.features(x)
        x = base.avgpool(x)
        
        feats = F.adaptive_avg_pool2d(x, (1, 1))
        feats = feats.squeeze(3).squeeze(2)
        
        x = torch.flatten(x, 1)
        logits = base.classifier(x)
        
        return logits, (feats, torch.zeros(feats.shape[0], 1))
        

class Jalal_features(Base_override):
    def __init__(self, base_model):
        super().__init__(base_model)


def _load_imagenet_model(model_details):
    base_model = models.__dict__[model_details['architecture']](pretrained=model_details['cnn_pretrained'],
                                                                num_classes=model_details['num_classes'])
    if not model_details['cnn_pretrained']:
        base_model.load_state_dict(model_details['state_dict'])

    return base_model


def _load_mnist_model(model_details):
    base_model = jalal_models.__dict__[model_details['architecture']](pretrained=model_details['cnn_pretrained'])

    if not model_details['cnn_pretrained']:
        base_model.load_state_dict(model_details['state_dict'])

    return base_model


def construct_Cnn_features(model_details):    
    if 'resnet' in model_details['architecture']:
        base_model = _load_imagenet_model(model_details)
        features = ResNet_features(base_model)
        
    elif "densenet" in model_details['architecture']:
        base_model = _load_imagenet_model(model_details)
        features = DenseNet_features(base_model)
        
    elif "vgg" in model_details['architecture']:
        base_model = _load_imagenet_model(model_details)
        features = VGG_features(base_model)
    
    elif "jalal" in model_details['architecture']:
        base_model = _load_mnist_model(model_details)
        features = Jalal_features(base_model)

        
    return features
