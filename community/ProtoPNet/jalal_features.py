import torch
import torch.nn as nn
import torch.nn.functional as F
import os

basepath = os.path.join(*os.path.split(os.path.realpath(__file__))[:-1])
model_file = os.path.join(basepath, "debug_mnist/mnist_epoch-5_seed-9.ckpt")


# Modified to accept 3 channel MNIST
class MNISTFeatures(nn.Module):
    def __init__(self, init_weights=True, batch_norm=False):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # conv1 -> M -> conv2 -> M
        self.kernel_sizes = [5, 2, 5, 2]
        # conv default to 1
        self.strides = [1, 2, 1, 2]
        self.paddings = [2, 0, 2, 0]
        self.n_layers = 2
        
        self.batch_norm = batch_norm
        if batch_norm:
            self.conv_block1 = nn.Sequential(
                *[conv1, nn.BatchNorm2d(32), nn.ReLU(inplace=True)]
            )
            self.conv_block2 = nn.Sequential(
                *[conv2, nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            )
        else:
            self.conv_block1 = nn.Sequential(
                *[conv1, nn.ReLU(inplace=True)]
            )
            self.conv_block2 = nn.Sequential(
                *[conv2, nn.ReLU(inplace=True)]
            )
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings
    

    def num_layers(self):
        '''
        the number of conv layers in the network
        '''
        return self.n_layers

    def __repr__(self):
        template = f'MNIST{self.num_layers()}, batch_norm={self.batch_norm}'
        return template
    
    def forward(self, x):
        out = self.conv_block1(x)
        out = self.maxpool1(out)
        out = self.conv_block2(out)
        out = self.maxpool2(out)
        return out


class MNISTClassifier(nn.Module):
    def __init__(self, features, init_weights=True):
        super().__init__()
        self.features = features
        self.fc = nn.Linear(3136, 1024)
        self.final_fc = nn.Linear(1024, 10)
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.shape[0], -1)
        out = F.relu(self.fc(out))
        return self.final_fc(out)


def _build_features(pretrained, batch_norm):
    init_weights = True
    if pretrained:
        init_weights = False

    model = MNISTFeatures(init_weights=init_weights, batch_norm=batch_norm)
    
    if pretrained:
        my_dict = torch.load(model_file)

        keys_to_remove = set()
        for key in my_dict:
            if key.startswith('fc') or key.startswith('final_fc'):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict, strict=False)
    
    return model


def jalal_features(pretrained=False):
    return _build_features(pretrained, batch_norm=False)


def jalal_bn_features(pretrained=False):
    return _build_features(pretrained, batch_norm=True)
