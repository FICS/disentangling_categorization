import torch
import torch.nn as nn

# from torchvision.transforms import Normalize
# import torchvision.models as models

# batch normalize for pytorch 1.5.1
def normalize(x, mean, std):
    # print('normalize')
    if mean == '' or mean == None:
        return x
        
    mean = torch.tensor(mean).cuda().view(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, 1, 1)
    std = torch.tensor(std).cuda().view(1, x.shape[1], 1, 1).repeat(x.shape[0], 1, 1, 1)
    xp = (x - mean) / std
    return xp

    
class ProtoWrapper(nn.Module):
    def __init__(self, pretrained, pretrained_multi, mean, std):
        super(ProtoWrapper, self).__init__()
        self.model = pretrained
        self.model_multi = pretrained_multi
        self.max_batch_size = 128
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        if len(x.shape) == 5:
            # batch the inputs from Batch and Distractors
            B, D = x.shape[:2]
            xp = x.reshape((B*D, *x.shape[2:]))
            logits = None
            proto_sos = None
            min_distances = None
            structures = None
            
            for start in range(0, B*D, self.max_batch_size):
                end = min(start+self.max_batch_size, B*D)
                logits_i, (proto_sos_i, structure_i), min_dist_i = self.forward_batch(xp[start:end])
                if logits is None:
                    logits = torch.zeros((B*D, *logits_i.shape[1:])).cuda()
                    proto_sos = torch.zeros((B*D, *proto_sos_i.shape[1:])).cuda()
                    min_distances = torch.zeros((B*D, *min_dist_i.shape[1:])).cuda()
                    structures = torch.zeros((B*D, *structure_i.shape[1:])).cuda()
                    
                logits[start:end] = logits_i
                proto_sos[start:end] = proto_sos_i
                min_distances[start:end] = min_dist_i
                structures[start:end] = structure_i
            
            # logits, proto_sos, min_distances = self.forward_batch(xp)
            
            proto_sos = proto_sos.reshape((B, D, *proto_sos.shape[1:]))
            logits = logits.reshape((B, D, logits.shape[1]))
            min_distances = min_distances.reshape((B, D, min_distances.shape[1]))
            structures = structures.reshape((B, D, structures.shape[1]))
            
        else:
            logits, (proto_sos, structures), min_distances = self.forward_batch(x)
            
        return logits, (proto_sos, structures), min_distances
        
    def forward_batch(self, x):
        x = normalize(x, self.mean, self.std)
        max_dist = (self.model.prototype_shape[1]
                    * self.model.prototype_shape[2]
                    * self.model.prototype_shape[3])
        # with torch.no_grad():
        logits, min_distances = self.model_multi(x)
        protoL_input, proto_distances = self.model.push_forward(x)
        # global prototype list
        prototype_activations = self.model.distance_2_similarity(min_distances)
        # local prototype list (batch under test)
        prototype_activation_patterns = self.model.distance_2_similarity(proto_distances)
        if self.model.prototype_activation_function == 'linear':
            prototype_activations = prototype_activations + max_dist
            prototype_activation_patterns = prototype_activation_patterns + max_dist
        
        # TODO: maybe not necessary to be in probas
        # prototype_activations = prototype_activations.softmax(dim=1)
        # eos = torch.zeros_like(prototype_activations[:, 0]).unsqueeze(1)
        
        # zero prob of EoS
        # eos[:, 0] = 0
        
        # B x P+1 = B x V
        # prototype_sos = torch.cat([eos, prototype_activations], dim=1)
        # debug('pa', prototype_sos.shape)
        # debug(prototype_sos[0])
        
        return logits, (prototype_activations, prototype_activations), min_distances
    
    def prelinguistic(self, x):
        _, sender_repr, _ = self.forward(x)    
        return sender_repr
    
    def choose_grad(self, mode, log=print):
        options = ['joint', 'last_only', 'warm', 'off']
        assert mode in options, f"Only support modes: {options}"
        
        model = self.model_multi
        if mode == 'last_only':
            for p in model.module.features.parameters():
                p.requires_grad = False
            for p in model.module.add_on_layers.parameters():
                p.requires_grad = False
            model.module.prototype_vectors.requires_grad = False
            for p in model.module.last_layer.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: last layer')


        if mode == 'warm':
            for p in model.module.features.parameters():
                p.requires_grad = False
            for p in model.module.add_on_layers.parameters():
                p.requires_grad = True
            model.module.prototype_vectors.requires_grad = True
            for p in model.module.last_layer.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: warm')


        if mode == 'joint':
            for p in model.module.features.parameters():
                p.requires_grad = False
            for p in model.module.add_on_layers.parameters():
                p.requires_grad = True
            model.module.prototype_vectors.requires_grad = True
            for p in model.module.last_layer.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: joint')
            
        if mode == 'off':
            for p in model.module.features.parameters():
                p.requires_grad = False
            for p in model.module.add_on_layers.parameters():
                p.requires_grad = False
            model.module.prototype_vectors.requires_grad = False
            for p in model.module.last_layer.parameters():
                p.requires_grad = False

            log(f'\t{self.__class__.__name__} configuration: off')


class ProtoBWrapper(ProtoWrapper):
    def __init__(self, pretrained, pretrained_multi, mean, std):
        super(ProtoBWrapper, self).__init__(pretrained, pretrained_multi, mean, std)
        
        
# class MultiHeadProtoWrapper(ProtoWrapper):
#     def __init__(self, pretrained, pretrained_multi, mean, std, h):
#         super(MultiHeadProtoWrapper, self).__init__(pretrained, pretrained_multi, mean, std)
#         self.heads = h
#         self.num_prototypes = self.model.num_prototypes
#         self.toheads = Variable(torch.rand(self.num_prototypes, self.num_prototypes * h), requires_grad=True)
        
#     def forward_batch(self, x):
#         x = normalize(x, self.mean, self.std)
#         # with torch.no_grad():
#         logits, min_distances = self.model_multi(x)
#         protoL_input, proto_distances = self.model.push_forward(x)
#         # global prototype list
#         prototype_activations = self.model.distance_2_similarity(min_distances)
#         # local prototype list (batch under test)
#         prototype_activation_patterns = self.model.distance_2_similarity(proto_distances)
#         if self.model.prototype_activation_function == 'linear':
#             prototype_activations = prototype_activations + max_dist
#             prototype_activation_patterns = prototype_activation_patterns + max_dist
        
#         # multi head structure
#         structures = torch.bmm(prototype_activations, self.toheads.softmax(dim=0))
#         # B x h x k
#         structures = structures.reshape(-1, self.heads, self.num_prototypes)
        
#         return logits, (prototype_activations, structures), min_distances
    
        
class ProtoWrapper2(ProtoWrapper):
    def __init__(self, pretrained, pretrained_multi, mean, std, topk=10):
        super(ProtoWrapper2, self).__init__(pretrained, pretrained_multi, mean, std)
        self.topk = topk
        self.prototype_vector_shape = self.model.prototype_shape[1]
        
    def forward(self, x):
        if len(x.shape) == 5:
            # batch the inputs from Batch and Distractors
            B, D = x.shape[:2]
            xp = x.reshape((B*D, *x.shape[2:]))
            logits = None
            proto_sos = None
            min_distances = None
            
            for start in range(0, B*D, self.max_batch_size):
                end = min(start+self.max_batch_size, B*D)
                logits_i, topk_proto_i, min_dist_i = self.forward_batch(xp[start:end])
                if logits is None:
                    logits = torch.zeros((B*D, *logits_i.shape[1:])).cuda()
                    topk_proto = torch.zeros((B*D, self.topk, self.prototype_vector_shape)).cuda()
                    min_distances = torch.zeros((B*D, *min_dist_i.shape[1:])).cuda()
                    
                logits[start:end] = logits_i
                topk_proto[start:end] = topk_proto_i
                min_distances[start:end] = min_dist_i
            
            # logits, proto_sos, min_distances = self.forward_batch(xp)
            
            topk_proto = topk_proto.reshape((B, D, self.topk, self.prototype_vector_shape))
            logits = logits.reshape((B, D, logits.shape[1]))
            min_distances = min_distances.reshape((B, D, min_distances.shape[1]))
            
        else:
            logits, topk_proto, min_distances = self.forward_batch(x)
            
        return logits, topk_proto, min_distances
        
    def forward_batch(self, x):
        x = normalize(x, self.mean, self.std)
        # with torch.no_grad():
        logits, min_distances = self.model_multi(x)
        protoL_input, proto_distances = self.model.push_forward(x)
        # global prototype list
        prototype_activations = self.model.distance_2_similarity(min_distances)
        # local prototype list (batch under test)
        prototype_activation_patterns = self.model.distance_2_similarity(proto_distances)
        if self.model.prototype_activation_function == 'linear':
            max_dist = (self.model.prototype_shape[1]
                        * self.model.prototype_shape[2]
                        * self.model.prototype_shape[3])

            prototype_activations = prototype_activations + max_dist
            prototype_activation_patterns = prototype_activation_patterns + max_dist
        
        prototype_activations = prototype_activations.softmax(dim=1)
        
        # B x softmax(act) -> B x topK
        ### TODO: Sorted by default
        _, idx = torch.topk(prototype_activations, self.topk, dim=1)
        # B x topk x P
        topk_prototypes = self.model.prototype_vectors[idx].squeeze(-1).squeeze(-1)
        
        return logits, topk_prototypes, min_distances
    
    def prelinguistic(self, x):
        _, topk_proto, _ = self.forward(x)    
        return topk_proto
    

# class CnnModel(nn.Module):
#     def __init__(self, model_str, option: str="", pretrained=True):
#         super(CnnModel, self).__init__()
#         self.model_str = model_str
#         # self.cnn = models.__dict__[f'{model_str}'](pretrained=pretrained).eval()
#         # not possible since we are re-training
#         # if option == 'relu7':
#         #     assert 'vgg' in model, f'Option {option} only applicable to VGG family!'
#         #     self.prelinguistic = nn.Sequential(*self.cnn.features[0:17])
#         # 
        
#         # self.features = self.cnn.features
#         self.features = base_architecture_to_features[model_str](pretrained=pretrained)
#         # B x E x 7 x 7 -> B x E x 1 x 1
#         self.add_on_layers = nn.AdaptiveAvgPool2d((1, 1))
#         # B x E x 1 x 1 -> B x E
#         # self.prelinguistic_pool = nn.AdaptiveAvgPool2d((1, 1))
#         # final term affected by agents
#         # return logits, handle softmax in loss
#         # self.classifier = self.cnn.classifier
        
#     def forward(self, x):
#         feats = self.features(x)
#         # B x E x H1 x W1
#         # resnet/vgg: H1=W1=7
#         # wx = self.add_on_layers(feats)
#         # wx = torch.flatten(wx, 1)
#         # logits = self.classifier(wx)
#         # encoding = self.add_on_layers(feats)
#         # B x E x 1 x 1
#         encoding = torch.flatten(encoding, 1)
#         # B x E
#         # return 3-tuple to be like Prototype model
#         return None, encoding, None
        

class CnnWrapper(nn.Module):
    """
    ImageNet weights (torchvision)
    """
    def __init__(self, model, model_multi, mean, std):
        super(CnnWrapper, self).__init__()
        self.model = model
        self.model_multi = model_multi
        self.mean = mean
        self.std = std
        self.max_batch_size = 128
        # statement = f"Using model {model}"
        # statement += f" with option {option}." if option != "" else "."
        # print(statement)
    
    def forward_batch(self, x):
        # print(x.shape, x.min(), x.max())
        x = normalize(x, self.mean, self.std)
        logits, (feats, structures) = self.model_multi.module.forward_feats(x)
        return logits, (feats, structures), None
    
    def forward(self, x):
        if len(x.shape) == 5:
            # process batch of distractors
            B, D = x.shape[:2]
            xp = x.reshape((B*D, *x.shape[2:]))
            
            logits = None
            encoding = None
            structures = None
            
            for start in range(0, B*D, self.max_batch_size):
                end = min(start+self.max_batch_size, B*D)
                logits_i, (encoding_i, structures_i), _ = self.forward_batch(xp[start:end])
                if encoding is None:
                    logits = torch.zeros((B*D, *logits_i.shape[1:])).cuda()
                    encoding = torch.zeros((B*D, *encoding_i.shape[1:])).cuda()
                    structures = torch.zeros((B*D, *structures_i.shape[1:])).cuda()
                    
                logits[start:end] = logits_i
                encoding[start:end] = encoding_i
                structures[start:end] = structures_i
                
            encoding = encoding.reshape((B, D, *encoding.shape[1:]))
            logits = logits.reshape((B, D, *logits.shape[1:]))
            structures = structures.reshape((B, D, *structures.shape[1:]))
            
            return logits, (encoding, structures), None
                
        else:
            return self.forward_batch(x)
        
    def prelinguistic(self, x):
        # give encoding
        _, sender_repr, _ = self.forward(x)
        return sender_repr
    
    def choose_grad(self, mode, log=print):
        options = ['joint', 'last_only', 'warm', 'off']
        assert mode in options, f"Only support modes: {options}"
        
        model = self.model_multi
        if mode == 'last_only':
            for p in model.module.base_model.parameters():
                p.requires_grad = False
            for p in model.module.classifier.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: last layer')

        if mode == 'warm':
            for p in model.module.base_model.parameters():
                p.requires_grad = False
            for p in model.module.classifier.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: warm')

        if mode == 'joint':
            for p in model.module.base_model.parameters():
                p.requires_grad = True
            for p in model.module.classifier.parameters():
                p.requires_grad = True

            log(f'\t{self.__class__.__name__} configuration: joint')
            
        if mode == 'off':
            for p in model.module.base_model.parameters():
                p.requires_grad = False
            for p in model.module.classifier.parameters():
                p.requires_grad = False

            log(f'\t{self.__class__.__name__} configuration: off')

            
class CnnBWrapper(CnnWrapper):
    """
    Load dataset weights, not ImageNet
    """
    def __init__(self, model, model_multi, mean, std):
        super(CnnBWrapper, self).__init__(model, model_multi, mean, std)
        
        
class CwWrapper(CnnWrapper):
    """
    Basic wrapper around CW model feature layer
    """
    def __init__(self, model, model_multi, mean, std):
        super(CwWrapper, self).__init__(model, model_multi, mean, std)
