import torch
import torch.nn as nn
import os
import numpy as np

from tqdm import tqdm

from community.ProtoPNet.settings import coefs
from losses import proto_losses, proto_weighting
from util import EpochHistory, process_exchange


def semiotic_social_epoch(state, epoch, device, train_loader, 
                          sender_percept, receiver_percept, game, 
                          optimizer, approach, log=print):
    epoch_log = EpochHistory(epoch)
    nb = train_loader.n_batches_per_epoch
    
    # sender_wrapper.model_multi.train()
    game.sender.train()
    game.receiver.train()
    
    if approach == 'cw' and len(state['semiotic_sgd_epochs']):
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

        conceptdir_train = state['concept_train_dir']
        conceptdir_test = state['concept_test_dir']
        concept_loaders = [
        torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=state['sender_mean'], std=state['sender_std']),
            ])),
            batch_size=state['train_batch_size'], shuffle=True,
            num_workers=4, pin_memory=False
        )
        for concept in state['concepts']
    ]

    
    with tqdm(total=nb) as pb:
        epoch_rolling_acc = []
        epoch_rolling_concept_acc = []
        epoch_rolling_loss = []
        
        for i, (sender_tuple, recv_targets, recv_tuple, sender_labels) in enumerate(train_loader):
            if train_loader.curr_epoch_mode == 'semiotic':
                # Get features feed-forward way
                sender_images = sender_tuple[0].to(device)
                concept_logits, (sender_feats, sender_structs), min_dists = sender_percept(sender_images)

                # with torch.no_grad():  # only message signal is allowed
                recv_images = recv_tuple[0].to(device)
                _, (recv_feats, recv_structs), _ = receiver_percept(recv_images)
            else:
                # Get features from loader cache
                concept_logits, (sender_feats, sender_structs), min_dists = None, sender_tuple, None
                _, (recv_feats, recv_structs), min_dists = None, recv_tuple, None

            recv_targets = recv_targets.to(device)
            sender_labels = sender_labels.to(device)
            
            sender_repr = (sender_feats.to(device), sender_structs)
            recv_repr = (recv_feats.to(device), recv_structs)
            social_loss, sender_message, receiver_output, aux_dict_i = game(sender_repr, 
                                                                            recv_targets, 
                                                                            receiver_input=recv_repr)
            optimizer.zero_grad()
            if train_loader.curr_epoch_mode == 'semiotic':
                if approach == 'proto':
                    # deal with proto losses
                    cross_ent, cluster_cost, separation_cost, l1 = proto_losses(
                        sender_percept, 
                        state['class_specific'], 
                        concept_logits, 
                        sender_labels,
                        min_dists
                    )
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    aux_dict_i['sign_cluster'] = cluster_cost.item()
                    aux_dict_i['sign_sep'] = separation_cost.item()
                    aux_dict_i['sign_l1'] = l1.item()

                    proto_loss = proto_weighting(
                        state['class_specific'],
                        cross_ent,
                        cluster_cost,
                        separation_cost,
                        l1,
                        coefs
                    )
                    semiotic_loss = state['sign_coef'] * proto_loss + state['social_coef'] * social_loss
                    loss_i = semiotic_loss
                elif approach == 'cw':
                    # update rotation matrix on certain time step
                    if (i + 1) % 30 == 0: 
                        sender_percept.model_multi.eval()
                        # access CW wrapper around base model weights
                        cw_model = sender_percept.model_multi.module.cw_model
                        with torch.no_grad():
                            # update the gradient matrix G
                            for concept_index, concept_loader in enumerate(concept_loaders):
                                cw_model.change_mode(concept_index)
                                for j, (X, _) in enumerate(concept_loader):
                                    X_var = torch.autograd.Variable(X).cuda()
                                    _ = sender_percept.model_multi(X_var)
                                    break
                            cw_model.update_rotation_matrix()
                            # change to ordinary mode
                            cw_model.change_mode(-1)
                        sender_percept.model_multi.train()
                    
                    cross_ent = torch.nn.functional.cross_entropy(concept_logits, sender_labels)
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    endtoend_loss = state['sign_coef'] * cross_ent + state['social_coef'] * social_loss
                    loss_i = endtoend_loss
                else:
                    # end-to-end CNN baseline
                    cross_ent = torch.nn.functional.cross_entropy(concept_logits, sender_labels)
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    endtoend_loss = state['sign_coef'] * cross_ent + state['social_coef'] * social_loss
                    loss_i = endtoend_loss
            else:
                # static case, concept logits = None
                loss_i = social_loss
             
            loss_i.backward()
            optimizer.step()
            
            trunc_messages, last_recv = process_exchange(sender_message, receiver_output)

            acc_i = last_recv.argmax(dim=1) == recv_targets
            acc_i = torch.mean(acc_i.float()).item()
            epoch_rolling_acc.append(acc_i)
            
            if train_loader.curr_epoch_mode == 'semiotic':
                _, c_preds = torch.max(concept_logits, dim=1)
                c_acc_i = c_preds == sender_labels
                c_acc_i = torch.mean(c_acc_i.float()).item()
                epoch_rolling_concept_acc.append(c_acc_i)

            epoch_rolling_loss.append(loss_i.cpu().item())

            str_loss = f"{np.mean(epoch_rolling_loss):.3f}"
            str_acc = f"{np.mean(epoch_rolling_acc):.3f}"
            str_c_acc = f"{np.mean(epoch_rolling_concept_acc) if len(epoch_rolling_concept_acc) else 0.0:.3f}"

            epoch_log.update_aux_info(aux_dict_i)
            epoch_log.update_main_loss(loss_i.cpu().item())

            for key, value in aux_dict_i.items():
                aux_dict_i[key] = f"{aux_dict_i.get(key, 0.0):.3f}"

            pb.update(1)
            pb.set_postfix(loss=str_loss, acc=str_acc, c_acc=str_c_acc, **aux_dict_i)
            # print(sender_message.argmax(dim=2)[0]) 
            # break
    # log(f'Epoch {epoch}: train acc={str_acc}, loss={str_loss}\n')
        
    # reset dali iterator
    train_loader.reset()
        
    return epoch_log


def classifier_epoch(state, epoch, device, train_loader, wrapper, optimizer, experiment, log=print):
    epoch_log = EpochHistory(epoch)
    nb = train_loader.n_batches_per_epoch
    # proto_wrapper.model_multi.train()
    
    with tqdm(total=nb) as pb:
        epoch_rolling_concept_acc = []
        epoch_rolling_loss = []
        
        for i, (image, label) in enumerate(train_loader):
            images = image.cuda()
            target = label.cuda()
            
            concept_logits, _, min_dists = wrapper(images)   
            aux_dict_i = {}
            
            optimizer.zero_grad()
            
            if experiment == 'proto':
                # semiotic case, deal with proto losses
                cross_ent, cluster_cost, separation_cost, l1 = proto_losses(
                    wrapper, 
                    state['class_specific'], 
                    concept_logits, 
                    target,
                    min_dists
                )
                aux_dict_i['sign_xe'] = cross_ent.item()
                aux_dict_i['sign_cluster'] = cluster_cost.item()
                aux_dict_i['sign_sep'] = separation_cost.item()
                aux_dict_i['sign_l1'] = l1.item()

                proto_loss = proto_weighting(
                    state['class_specific'],
                    cross_ent,
                    cluster_cost,
                    separation_cost,
                    l1,
                    coefs
                )
                proto_loss.backward()
                optimizer.step()

                loss_i = proto_loss
            else:
                # end-to-end CNN baseline
                cross_ent = torch.nn.functional.cross_entropy(concept_logits, target)
                aux_dict_i['sign_xe'] = cross_ent.item()
                cross_ent.backward()
                optimizer.step()
                
                loss_i = cross_ent
             
            _, c_preds = torch.max(concept_logits, dim=1)
            c_acc_i = c_preds == target
            c_acc_i = torch.mean(c_acc_i.float()).item()
            epoch_rolling_concept_acc.append(c_acc_i)

            epoch_rolling_loss.append(loss_i.cpu().item())

            str_loss = f"{np.mean(epoch_rolling_loss):.3f}"
            str_c_acc = f"{np.mean(epoch_rolling_concept_acc) if len(epoch_rolling_concept_acc) else 0.0:.3f}"

            epoch_log.update_aux_info(aux_dict_i)
            epoch_log.update_main_loss(loss_i.cpu().item())

            for key, value in aux_dict_i.items():
                aux_dict_i[key] = f"{aux_dict_i.get(key, 0.0):.3f}"

            pb.update(1)
            pb.set_postfix(loss=str_loss, c_acc=str_c_acc, **aux_dict_i)
            # break
        
    # reset dali iterator
    train_loader.reset()
        
    return epoch_log
    

# def semiotic_reification_epoch(state, epoch, device, train_loader, 
#                                 sender_percept, receiver_percept, game, 
#                                 optimizer, experiment, log=print):
#     epoch_log = EpochHistory(epoch)
#     nb = train_loader.n_batches_per_epoch
    
#     # sender_wrapper.model_multi.train()
#     game.sender.train()
#     game.receiver.train()

#     with tqdm(total=nb) as pb:
#         epoch_rolling_acc = []
#         epoch_rolling_concept_acc = []
#         epoch_rolling_loss = []
        
#         for i, (sender_tuple, recv_targets, recv_tuple, sender_labels) in enumerate(train_loader):
#             sender_images = sender_tuple[0].to(device)

#             max_dist = (sender_percept.model.module.prototype_shape[1]
#                         * sender_percept.model.module.prototype_shape[2]
#                         * sender_percept.model.module.prototype_shape[3])
                        
#             logits, min_distances = sender_percept.model_multi(sender_images)
#             protoL_input, proto_distances = sender_percept.model.push_forward(sender_images)
#             # global prototype list
#             prototype_activations = sender_percept.model.distance_2_similarity(min_distances)
#             # local prototype list (batch under test)
#             prototype_activation_patterns = sender_percept.model.distance_2_similarity(proto_distances)
#             if sender_percept.model.prototype_activation_function == 'linear':
#                 prototype_activations = prototype_activations + max_dist
#                 prototype_activation_patterns = prototype_activation_patterns + max_dist

#             top_patterns = []
#             for activations_i in prototype_activations:
#                 array_act, sorted_indices_act = torch.sort(activations_i)
#                 top_patterns.append(prototype_activation_patterns[sorted_indices_act[-1].item()])

#             # ground truth for AE
#             top_patterns = torch.stack(top_patterns)
            

#     return


def semiotic_reification_epoch(state, epoch, device, train_loader, 
                              sender_percept, receiver_percept, game, 
                              optimizer, approach, log=print):
    epoch_log = EpochHistory(epoch)
    nb = train_loader.n_batches_per_epoch
    
    # sender_wrapper.model_multi.train()
    game.sender.train()
    game.receiver.train()
    
    if approach == 'cw' and len(state['semiotic_sgd_epochs']):
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets

        conceptdir_train = state['concept_train_dir']
        conceptdir_test = state['concept_test_dir']
        concept_loaders = [
        torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(conceptdir_train, concept), transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=state['sender_mean'], std=state['sender_std']),
            ])),
            batch_size=state['train_batch_size'], shuffle=True,
            num_workers=4, pin_memory=False
        )
        for concept in state['concepts']
    ]
    
    if epoch > state['apply_symbols_epoch']:
        game.receiver.apply_symbols = True
    
    with tqdm(total=nb) as pb:
        epoch_rolling_acc = []
        epoch_rolling_concept_acc = []
        epoch_rolling_loss = []
        
        for i, (sender_tuple, recv_targets, recv_tuple, sender_labels) in enumerate(train_loader):
            if train_loader.curr_epoch_mode == 'semiotic' or train_loader.curr_epoch_mode == 'reification':
                # Get features feed-forward way
                sender_images = sender_tuple[0].to(device)
                concept_logits, (sender_feats, sender_structs), min_dists = sender_percept(sender_images)

                # with torch.no_grad():  # only message signal is allowed
                recv_images = recv_tuple[0].to(device)
                _, (recv_feats, recv_structs), _ = receiver_percept(recv_images)
            else:
                # Get features from loader cache
                concept_logits, (sender_feats, sender_structs), min_dists = None, sender_tuple, None
                _, (recv_feats, recv_structs), min_dists = None, recv_tuple, None

            recv_targets = recv_targets.to(device)
            sender_labels = sender_labels.to(device)
            
            sender_repr = (sender_feats, sender_structs)
            recv_repr = (recv_feats, recv_structs)
            social_loss, sender_message, receiver_output, aux_dict_i = game(sender_repr, 
                                                                            recv_targets, 
                                                                            receiver_input=recv_repr)

            try:
                smatches = game.sender.matches
            except AttributeError:
                smatches = torch.zeros([1, 1])

            try:
                rmatches = game.receiver.matches
            except AttributeError:
                rmatches = torch.zeros([1, 1])
            
            str_smatch = f"{smatches[0].detach().cpu().numpy()}"
            str_rmatch = f"{rmatches[0].detach().cpu().numpy()}"

            optimizer.zero_grad()
            if train_loader.curr_epoch_mode == 'semiotic':
                if approach == 'proto':
                    # deal with proto losses
                    cross_ent, cluster_cost, separation_cost, l1 = proto_losses(
                        sender_percept, 
                        state['class_specific'], 
                        concept_logits, 
                        sender_labels,
                        min_dists
                    )
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    aux_dict_i['sign_cluster'] = cluster_cost.item()
                    aux_dict_i['sign_sep'] = separation_cost.item()
                    aux_dict_i['sign_l1'] = l1.item()

                    proto_loss = proto_weighting(
                        state['class_specific'],
                        cross_ent,
                        cluster_cost,
                        separation_cost,
                        l1,
                        coefs
                    )
                    semiotic_loss = state['sign_coef'] * proto_loss + state['social_coef'] * social_loss
                    loss_i = semiotic_loss
                elif approach == 'cw':
                    # update rotation matrix on certain time step
                    if (i + 1) % 30 == 0: 
                        sender_percept.model_multi.eval()
                        # access CW wrapper around base model weights
                        cw_model = sender_percept.model_multi.module.cw_model
                        with torch.no_grad():
                            # update the gradient matrix G
                            for concept_index, concept_loader in enumerate(concept_loaders):
                                cw_model.change_mode(concept_index)
                                for j, (X, _) in enumerate(concept_loader):
                                    X_var = torch.autograd.Variable(X).cuda()
                                    _ = sender_percept.model_multi(X_var)
                                    break
                            cw_model.update_rotation_matrix()
                            # change to ordinary mode
                            cw_model.change_mode(-1)
                        sender_percept.model_multi.train()
                    
                    cross_ent = torch.nn.functional.cross_entropy(concept_logits, sender_labels)
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    endtoend_loss = state['sign_coef'] * cross_ent + state['social_coef'] * social_loss
                    loss_i = endtoend_loss
                else:
                    # end-to-end CNN baseline
                    cross_ent = torch.nn.functional.cross_entropy(concept_logits, sender_labels)
                    aux_dict_i['sign_xe'] = cross_ent.item()
                    endtoend_loss = state['sign_coef'] * cross_ent + state['social_coef'] * social_loss
                    loss_i = endtoend_loss
            else:
                # static case, concept logits = None
                loss_i = social_loss
             
            loss_i.backward()
            optimizer.step()
            
            trunc_messages, last_recv = process_exchange(sender_message, receiver_output)

            acc_i = last_recv.argmax(dim=1) == recv_targets
            acc_i = torch.mean(acc_i.float()).item()
            epoch_rolling_acc.append(acc_i)
            
            if train_loader.curr_epoch_mode == 'semiotic':
                _, c_preds = torch.max(concept_logits, dim=1)
                c_acc_i = c_preds == sender_labels
                c_acc_i = torch.mean(c_acc_i.float()).item()
                epoch_rolling_concept_acc.append(c_acc_i)

            epoch_rolling_loss.append(loss_i.cpu().item())

            str_loss = f"{np.mean(epoch_rolling_loss):.3f}"
            str_acc = f"{np.mean(epoch_rolling_acc):.3f}"
            str_c_acc = f"{np.mean(epoch_rolling_concept_acc) if len(epoch_rolling_concept_acc) else 0.0:.3f}"

            epoch_log.update_aux_info(aux_dict_i)
            epoch_log.update_main_loss(loss_i.cpu().item())
            epoch_log.update_reconstructions(
                game.sender.recons[:, :, :, 0, 0].cpu().detach().numpy(), 
                game.receiver.recons[:, :, :, 0, 0].cpu().detach().numpy()
            )

            for key, value in aux_dict_i.items():
                aux_dict_i[key] = f"{aux_dict_i.get(key, 0.0):.3f}"

            pb.update(1)
            pb.set_postfix(loss=str_loss, acc=str_acc, c_acc=str_c_acc, smatch=str_smatch, rmatch=str_rmatch, **aux_dict_i)
        
    # reset dali iterator
    train_loader.reset()
        
    return epoch_log


# def semiotic_reification_epoch(state, epoch, train_loader, sender_percept, receiver_percept, game, optimizer):
#     nb = train_loader.n_batches_per_epoch
#     game.sender.train()
#     game.receiver.train()
    
#     with tqdm(total=nb) as pb:
#         for i, (sender_tuple, recv_targets, recv_tuple, sender_labels) in enumerate(train_loader):
#             sender_images = sender_tuple[0].to(device)
#             concept_logits, sender_repr, min_dists = sender_percept(sender_images)
            
#             recv_targets = recv_targets.to(device)
#             sender_labels = sender_labels.to(device)
            
#             # with torch.no_grad():  # only message signal is allowed
#             recv_images = recv_tuple[0].to(device)
#             _, recv_repr, _ = receiver_percept(recv_images)

#             loss, sender_message, receiver_output, aux_dict_i = game(sender_repr, 
#                                                                      recv_targets, 
#                                                                      receiver_input=recv_repr)
            
#             try:
#                 smatches = game.sender.matches
#             except AttributeError:
#                 smatches = torch.zeros([1, 1])

#             try:
#                 rmatches = game.receiver.matches
#             except AttributeError:
#                 rmatches = torch.zeros([1, 1])

                
#             msg_str = [str(i) for i in sender_message[0, :, :].argmax(dim=1).cpu().numpy()]
#             msg_str = ".".join(msg_str)
                
#             str_loss = f"{loss.cpu().item():.3f}"
#             str_smatch = f"{smatches[0].detach().cpu().numpy()}"
#             str_rmatch = f"{rmatches[0].detach().cpu().numpy()}"
#             str_label = f"{sender_labels[0].detach().cpu().item()}"
#             # system_loss, aux_dict = reconstruction_loss(top_patterns, message, None, receiver_output, None)
#             # system_loss = system_loss.mean()
#             # str_sys = f"{system_loss.cpu().item():.3f}"
            
#             # loss = 0.5*loss + 0.5*system_loss
            
#             optimizer.zero_grad()
            
#             loss.backward()
#             optimizer.step()
            
#             pb.set_postfix(loss=str_loss, msg=msg_str, smatch=str_smatch, rmatch=str_rmatch, label=str_label, **aux_dict_i)
#             pb.update(1)