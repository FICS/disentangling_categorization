import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import construct_CW


class Base_override(nn.Module):
    """
    Base class for overriding ConceptWhitening models for feature output. 
    """
    def __init__(self, base_model, model_details):
        super(Base_override, self).__init__()
        self.concepts = model_details['concepts']
        self.cw_model = base_model
        self.base_model = base_model.model
        self.k = len(self.concepts)
        
    def forward_feats(self, x):
        # logits, (None feats, None structure)
        return self.base_model(x), (None, None)
    
    def forward(self, x):
        logits, x = self.forward_feats(x)
        return logits
        

class ResNet_features(Base_override):
    def __init__(self, base_model, model_details):
        super(ResNet_features, self).__init__(base_model, model_details)
        self.classifier = self.base_model.fc
        self.structures = []
        
        # repr for resnet_cw model from ConceptWhitening/plot_functions.py
        def hook(module, input, output):
            from ConceptWhitening.MODELS.iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            #print(size_X)
            X_hat = X_hat.view(*size_X)
            
            # print(X_hat.shape)
            # print(X_hat.sum((2,3)).shape)
            self.structures.append(X_hat.sum((2,3))[:, :self.k])
            
        layer = int(self.cw_model.whitened_layers[-1])
        layers = self.cw_model.layers
        print('layers' , layers, 'select', layer)
        if layer <= layers[0]:
            self.base_model.layer1[layer-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1]:
            self.base_model.layer2[layer-layers[0]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2]:
            self.base_model.layer3[layer-layers[0]-layers[1]-1].bn1.register_forward_hook(hook)
        elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
            self.base_model.layer4[layer-layers[0]-layers[1]-layers[2]-1].bn1.register_forward_hook(hook)
            
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    """
    def forward_feats(self, x):
        # update from hooks
        self.structures = []
        
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
        
        structures = torch.cat(self.structures, dim=1)
        return logits, (x, structures)
    
    
class DenseNet_features(Base_override):
    def __init__(self, base_model, model_details):
        super(DenseNet_features, self).__init__(base_model, model_details)
        self.classifier = self.base_model.classifier
        self.structures = []
        
        # repr for resnet_cw model from ConceptWhitening/plot_functions.py
        def hook(module, input, output):
            from ConceptWhitening.MODELS.iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            #print(size_X)
            X_hat = X_hat.view(*size_X)
            
            # print(X_hat.shape)
            # print(X_hat.sum((2,3)).shape)
            self.structures.append(X_hat.sum((2,3))[:, :self.k])
            
        layer = int(self.cw_model.whitened_layers[-1])
        print('select', layer)
        # see L48-L55 of ConceptWhitening/MODEL/model_resnet.py
        if layer == 1:
            self.base_model.features.norm0.register_forward_hook(hook)
        elif layer == 2:
            self.base_model.features.transition1.norm.register_forward_hook(hook)
        elif layer == 3:
            self.base_model.features.transition2.norm.register_forward_hook(hook)
        elif layer == 4:
            self.base_model.features.transition3.norm.register_forward_hook(hook)
        elif layer == 5:
            self.base_model.features.norm5.register_forward_hook(hook)
            
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html
    """
    def forward_feats(self, x):
        # update from hooks
        self.structures = []
        
        base = self.base_model
        
        features = base.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        logits = base.classifier(out)
        
        structures = torch.cat(self.structures, dim=1)
        return logits, (out, structures)
    
    
class VGG_features(Base_override):
    def __init__(self, base_model, model_details):
        super(VGG_features, self).__init__(base_model, model_details)
        self.classifier = self.base_model.classifier[6]
        self.structures = []
        
        # repr for resnet_cw model from ConceptWhitening/plot_functions.py
        def hook(module, input, output):
            from ConceptWhitening.MODELS.iterative_normalization import iterative_normalization_py
            #print(input)
            X_hat = iterative_normalization_py.apply(input[0], module.running_mean, module.running_wm, module.num_channels, module.T,
                                                    module.eps, module.momentum, module.training)
            size_X = X_hat.size()
            size_R = module.running_rot.size()
            X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

            X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, module.running_rot)
            #print(size_X)
            X_hat = X_hat.view(*size_X)
            
            # print(X_hat.shape)
            # print(X_hat.sum((2,3)).shape)
            self.structures.append(X_hat.sum((2,3))[:, :self.k])
            
        layer = int(self.cw_model.whitened_layers[-1])
        layers = self.cw_model.layers
        print('layers' , layers, 'select', layer-1)
            
        self.base_model.features[layers[layer-1]].register_forward_hook(hook)
            
    """
    https://pytorch.org/vision/stable/_modules/torchvision/models/vgg.html
    """
    def forward_feats(self, x):
        # update from hooks
        self.structures = []
        
        base = self.base_model
        
        x = base.features(x)
        x = base.avgpool(x)
        
        feats = F.adaptive_avg_pool2d(x, (1, 1))
        feats = feats.squeeze(3).squeeze(2)
        
        x = torch.flatten(x, 1)
        logits = base.classifier(x)
        
        structures = torch.cat(self.structures, dim=1)
        return logits, (feats, structures)
        

def construct_CW_features(model_details):
    # disable cnn checkpoint since we will load our own CW model
    model_details['base_cnn_ckpt'] = None
    base_model = construct_CW(model_details)
    base_model.load_state_dict(model_details['state_dict'])
    
    if model_details['architecture'] == "resnet50_cw":
        features = ResNet_features(base_model, model_details)
        
    elif model_details['architecture'] == "resnet18_cw":
        features = ResNet_features(base_model, model_details)

    elif model_details['architecture'] == "resnet50_baseline":
        features = ResNet_features(base_model, model_details)

    elif model_details['architecture'] == "resnet18_baseline":
        features = ResNet_features(base_model, model_details)

    elif model_details['architecture'] == "densenet161_cw":
        features = DenseNet_features(base_model, model_details)
        
    elif model_details['architecture'] == 'densenet161_baseline':
        features = DenseNet_features(base_model, model_details)

    elif model_details['architecture'] == "vgg16_bn_cw":
        features = VGG_features(base_model, model_details)
        
    elif model_details['architecture']== "vgg16_bn_baseline":
        features = VGG_features(base_model, model_details)

    return features
