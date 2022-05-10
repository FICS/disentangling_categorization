from .MODELS.model_resnet import *


def construct_CW(model_details):
    if model_details['architecture'] == "resnet50_cw":
        model = ResidualNetTransfer(model_details['num_classes'], 
                                    model_details['act_mode'], 
                                    model_details['whitened_layers'], 
                                    arch = 'resnet50', 
                                    layers = [3, 4, 6, 3], 
                                    model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == "resnet18_cw":
        model = ResidualNetTransfer(model_details['num_classes'], 
                                    model_details['act_mode'], 
                                    model_details['whitened_layers'], 
                                    arch = 'resnet18', 
                                    layers = [2, 2, 2, 2], 
                                    model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == "resnet50_baseline":
        model = ResidualNetBN(model_details['num_classes'], 
                              arch = 'resnet50', 
                              layers = [3, 4, 6, 3], 
                              model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == "resnet18_baseline":
        model = ResidualNetBN(model_details['num_classes'], 
                              arch = 'resnet18', 
                              layers = [2, 2, 2, 2],
                              model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == "densenet161_cw":
        model = DenseNetTransfer(model_details['num_classes'], 
                                 model_details['act_mode'], 
                                 model_details['whitened_layers'], 
                                 arch = 'densenet161', 
                                 model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == 'densenet161_baseline':
        model = DenseNetBN(model_details['num_classes'], 
                           arch = 'densenet161', 
                           model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture'] == "vgg16_bn_cw":
        model = VGGBNTransfer(model_details['num_classes'], 
                              model_details['act_mode'], 
                              model_details['whitened_layers'], 
                              arch = 'vgg16_bn', 
                              model_file=model_details['base_cnn_ckpt'])
    elif model_details['architecture']== "vgg16_bn_baseline":
        model = VGGBN(model_details['num_classes'], 
                      arch='vgg16_bn', 
                      model_file=model_details['base_cnn_ckpt']) #'vgg16_bn_places365.pt')
        
    return model