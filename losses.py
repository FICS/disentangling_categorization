import torch
import torch.nn as nn

from torch.nn import functional as F
from community.info_theory import gap_mi_first_second, calc_entropy


def Lp_reconstruction_loss(batch_images, output_images, absolute_error=True):
    batch_images = batch_images.view(-1, 64)
    output_images = output_images.view(-1, 64)
    if absolute_error:
        return torch.sum(torch.abs(batch_images - output_images), dim=1), {}
    else:
        return torch.sum(torch.mul(batch_images - output_images, batch_images - output_images), dim=1), {}

    
def loss_nll(_sender_input, _message, _receiver_input, receiver_output, labels):
    """
    NLL loss - differentiable and can be used with both GS and Reinforce
    """
    nll = F.nll_loss(receiver_output, labels, reduction="none")
    acc = (labels == receiver_output.argmax(dim=1)).float().mean()
    return nll, {'acc': acc}


def loss_xent(_sender_input,  _message, _receiver_input, receiver_output, _labels):
    loss = F.cross_entropy(receiver_output, _labels, reduction="none")
    return loss, {}


def system_loss(sender_input, receiver_input, receiver_output, receiver_hiddens, labels):
    return 0.


def least_effort(sender_input, step, message, receiver_input, receiver_output, labels, vocab_size):
    """
    Least Effort Pressure vocabulary loss as formulated in Luna et al. (2020). 
    
    sender_input: tuple of target (feats, structures)
    step: current time step $t$
    message: message across all time steps $L$
    receiver_input: distractor feats
    receiver_output: pre-decision space from receiver
    labels: index of target in distractors
    """
    B = receiver_output.size(0)
    message_prev = message[:, :step+1, ...]
    C_t = torch.zeros(receiver_output.size(0), vocab_size).to(receiver_output.device)
    # uniques = torch.unique(message_prev.argmax(dim=-1), dim=1)
    # C_t = uniques.size(1)
    uniques_count = torch.zeros(B)
                                                      
    for b in range(B):
        uniques = torch.unique(message_prev[b].argmax(dim=-1))
        # normalized discount
        C_t[b, :] = torch.numel(uniques) / vocab_size
        
    # word generated at this step
    w = message_prev[:, step, ...].argmax(dim=-1)
    
    # score
    # s_w = message[torch.arange(B), step, w]
    
    C_t = C_t[torch.arange(B), w] * (step + 1.0)
    
    # discount and exp
    # adjusted_scores = torch.exp(message[b, step, ...] - C_t)

    # loss[b] = C_t - s_w + torch.sum(adjusted_scores)
    # num = torch.exp(s_w - C_t[torch.arange(B), w])
    # den = torch.sum(torch.exp(message[:, step, ...] - C_t), dim=-1)
    # loss = -torch.log(num / den)

    # uniques_count[b] = C_t
        
    return C_t, {}


def symbolic_loss(sender_input, step, message, receiver_input, receiver_output, labels, vocab_size):
    feats, structures = sender_input
    return 0


def entropy(sender_input, message, receiver_input, receiver_output, labels, vocab_size):
    """
    Entropy loss.
    
    sender_input: target feats
    step: current time step $t$
    message: message across all time steps $L$
    receiver_input: distractor feats
    receiver_output: pre-decision space from receiver
    labels: index of target in distractors
    """
    return -calc_entropy(message), {}


class PosDis:
    def __init__(self, k):
        self.k = k
    
    def posdis(self, sender_input, message, receiver_input, receiver_output, labels, vocab_size):
        """
        Positional disentanglement metric, introduced in "Compositionality and Generalization in Emergent Languages",
        Chaabouni et al., ACL 2020.

        sender_input: target feats
        step: current time step $t$
        message: message across all time steps $L$
        receiver_input: distractor feats
        receiver_output: pre-decision space from receiver
        labels: index of target in distractors
        """
        _, attributes = torch.topk(sender_input, k=self.k)
        return -gap_mi_first_second(attributes, message.argmax(dim=-1)), {}

    
class BoSDis:
    def __init__(self, k):
        self.k = k
    
    def bosdis(self, sender_input, message, receiver_input, receiver_output, labels, vocab_size):
        """
        Bag of Symbols disentanglement metric, introduced in "Compositionality and Generalization in Emergent Languages",
        Chaabouni et al., ACL 2020.

        sender_input: target feats
        step: current time step $t$
        message: message across all time steps $L$
        receiver_input: distractor feats
        receiver_output: pre-decision space from receiver
        labels: index of target in distractors
        """
        batch_size = message.size(0)
        histogram = torch.zeros(batch_size, vocab_size, device=message.device)
        for v in range(vocab_size):
            histogram[:, v] = message.argmax(dim=-1).eq(v).sum(dim=-1)
        histogram = histogram[:, 1:]  # ignoring eos symbol
        
        _, attributes = torch.topk(sender_input, k=self.k)
        
        return -gap_mi_first_second(attributes, histogram), {}


def unpack_losses(aux_losses, aux_weights):
    # unpack aux losses and their meta data/weights
    aux = []
    sys = []
    for loss_str, pack in zip(aux_losses, aux_weights):
        if loss_str == 'least_effort':
            weight = pack
            aux.append((least_effort, weight))
        elif loss_str == 'posdis':
            k_value, weight = pack
            pd = PosDis(k=k_value)
            sys.append((pd.posdis, weight))
        elif loss_str == 'bosdis':
            k_value, weight = pack
            bd = BoSDis(k=k_value)
            sys.append((bd.bosdis, weight))
        elif loss_str == 'entropy':
            weight = pack
            sys.append((entropy, weight))
        elif loss_str == 'symbolic_loss':
            weight = pack
            aux.append((symbolic_loss, weight))
        elif loss_str == 'Lp_reconstruction_loss':
            weight = pack
            aux.append((Lp_reconstruction_loss, weight))
        else:
            raise ValueError(f"Unsupported aux. loss: {loss_str}")
    
    return aux, sys


def proto_losses(wrapper, class_specific, output, labels, min_distances, use_l1_mask=False):
    labels_int = labels.cpu().detach().numpy().astype(int)
    
    cross_entropy = torch.nn.functional.cross_entropy(output, labels)

    model = wrapper.model_multi
    if class_specific:
        max_dist = (model.module.prototype_shape[1]
                    * model.module.prototype_shape[2]
                    * model.module.prototype_shape[3])

        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        # calculate cluster cost
        prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,labels_int]).cuda()
        inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
        cluster_cost = torch.mean(max_dist - inverted_distances)

        # calculate separation cost
        prototypes_of_wrong_class = 1 - prototypes_of_correct_class
        inverted_distances_to_nontarget_prototypes, _ = \
            torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
        separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

        # calculate avg cluster cost
        avg_separation_cost = \
            torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
        avg_separation_cost = torch.mean(avg_separation_cost)

        if use_l1_mask:
            l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
        else:
            l1 = model.module.last_layer.weight.norm(p=1) 

    else:
        min_distance, _ = torch.min(min_distances, dim=1)
        cluster_cost = torch.mean(min_distance)
        l1 = model.module.last_layer.weight.norm(p=1)
        separation_cost = 0.0
        
    return cross_entropy, cluster_cost, separation_cost, l1


def proto_weighting(class_specific, cross_entropy, cluster_cost, separation_cost, l1, coefs=None):
    if class_specific:
        if coefs is not None:
            proto_loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['sep'] * separation_cost
                        + coefs['l1'] * l1)
        else:
            proto_loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
    else:
        if coefs is not None:
            proto_loss = (coefs['crs_ent'] * cross_entropy
                        + coefs['clst'] * cluster_cost
                        + coefs['l1'] * l1)
        else:
            proto_loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            
    return proto_loss
