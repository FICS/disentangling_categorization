import torch
import numpy as np

from tqdm import tqdm
from util import process_exchange
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Signaling accuracy
def semiotic_signal_test(sender_wrapper, game,
                         sender_interpret, receiver_interpret, 
                         loader, device, return_metrics=False):
    game.sender.eval()
    game.receiver.eval()
    sender_wrapper.model_multi.eval()
    
    concept_actuals = []
    concept_preds = []
    
    actuals = []
    preds = []
    
    nb = loader.n_batches_per_epoch
            
    with tqdm(total=nb) as pb:
        with torch.no_grad():
            for i, (sender_tuple, recv_targets, recv_tuple, sender_labels) in enumerate(loader):
                if loader.curr_epoch_mode == 'semiotic' or loader.curr_epoch_mode == 'reification':
                    # Get features feed-forward way
                    sender_images = sender_tuple[0].to(device)
                    concept_logits, (sender_feats, sender_structs), min_dists = sender_interpret(sender_images)

                    # with torch.no_grad():  # only message signal is allowed
                    recv_images = recv_tuple[0].to(device)
                    _, (recv_feats, recv_structs), _ = receiver_interpret(recv_images)
                else:
                    # Get features from loader cache
                    concept_logits, (sender_feats, sender_structs), min_dists = None, sender_tuple, None
                    _, (recv_feats, recv_structs), min_dists = None, recv_tuple, None

                recv_targets = recv_targets.to(device)
                sender_labels = sender_labels.to(device)

                sender_repr = (sender_feats.to(device), sender_structs)
                recv_repr = (recv_feats.to(device), recv_structs)
                _, sender_message, receiver_output, _ = game(sender_repr, 
                                                             recv_targets, 
                                                             receiver_input=recv_repr)
                trunc_messages, last_recv = process_exchange(sender_message, receiver_output)
                preds_i = last_recv.argmax(dim=1) 
                preds_i = np.asarray(preds_i.cpu())

                preds.extend(list(preds_i))
                actuals.extend(list(recv_targets.detach().cpu().numpy()))
                
                if concept_logits is not None:
                    _, cpreds = torch.max(concept_logits, dim=1)
                    cactuals = sender_labels
                    
                    concept_preds.extend(list(cpreds.detach().cpu().numpy()))
                    concept_actuals.extend(list(cactuals.detach().cpu().numpy()))

                pb.update(1)
                # break
                
            loader.reset()
    
    metrics = {}
    
    acc = accuracy_score(actuals, preds)
    metrics['accuracy'] = acc
    
    if len(concept_actuals) > 0:
        concept_acc = accuracy_score(concept_actuals, concept_preds)
        metrics['concept_accuracy'] = concept_acc
        
    if return_metrics:
        return metrics