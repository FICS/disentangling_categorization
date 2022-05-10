import torch
import numpy as np
import logging
import os

import util
import train
import optim
import community.ProtoPNet.push as push

from community.ConvNet import test_and_train as ct
from community import train_and_test as tnt
from community.ProtoPNet.preprocess import preprocess_input_function
from util import pickle_write, pickle_load, EpochHistory
from optim import set_eval, set_train, disable_parameter_requires_grad, log_whole_system_params
from test_tasks import semiotic_signal_test


# Learn social task
# ==================================
def semiotic_signal_task(state, start_epoch, epoch_histories,
                         semiotic_train_loader, semiotic_test_loader, 
                         train_push_loader, ll_train_loader, ll_test_loader,
                         sender_percept, recv_percept, signal_game,
                         static_optimizer, semiotic_optimizer, classifier_optimizer,
                         device, train_fn=train.semiotic_social_epoch):
    log = logging.getLogger('tasks/semiotic_social')
    approach = state['approach']
    history_path = state['history_path']
    img_dir = os.path.join(state['save_dir'], 'sign-img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # housekeeping
    # ==================================
    recv_percept.model_multi.eval()
    
    # Check Epoch 0 metrics using cached representations
    epoch_log = EpochHistory(start_epoch)
    metrics = semiotic_signal_test(sender_percept, signal_game, 
                                   sender_percept, recv_percept, 
                                   semiotic_test_loader, device, return_metrics=True)
    epoch_log.log_accuracy(metrics['accuracy'])

    torch.save(signal_game.sender.state_dict(), os.path.join(state['save_dir'], f'sender_e{start_epoch}.pth'))
    torch.save(signal_game.receiver.state_dict(), os.path.join(state['save_dir'], f'receiver_e{start_epoch}.pth'))
    log.info(f"Checkpointed at {state['save_dir']}")

    # baseline sender classifier accuracy
    if approach == 'proto':
        pack = tnt.test(model=sender_percept.model_multi, dataloader=ll_test_loader,
                        class_specific=True, log=log.info)
        epoch_log.log_concept_accuracies({'push': pack[2]})
        epoch_log.log_concept_costs({'push': pack})
    else:
        accu = ct.test(state, sender_percept.model_multi, ll_test_loader, log.info)
        epoch_log.log_concept_accuracies({'push': accu}) 
        
    epoch_histories.append(epoch_log)

    # start
    # ==================================

    for epoch in np.arange(start_epoch+1, state['epochs']+1, 1):
        # decide the loader/optim to use based on this epoch's task
        semiotic_sgd = epoch >= state['semiosis_start'] and epoch in state['semiotic_sgd_epochs']
        semiotic_push = epoch >= state['semiosis_start'] and epoch in state['semiotic_push_epochs']
        # If we do push, assume semitotic sgd scenario
        semiotic_sgd = (semiotic_sgd or semiotic_push)
        
        train_loader = semiotic_train_loader
        test_loader = semiotic_test_loader

        if semiotic_sgd or semiotic_push:
            _optim = semiotic_optimizer
            epoch_mode = 'semiotic'
        else:
            _optim = static_optimizer
            epoch_mode = 'static'

        # disable grad and only enable next
        for model in [sender_percept, signal_game.sender, recv_percept, signal_game.receiver]:
            disable_parameter_requires_grad(model)

        set_eval([sender_percept, recv_percept])
    
        # init loader based on task (for caching)
        train_loader.start_epoch(epoch_mode)
        test_loader.start_epoch(epoch_mode)

        # enable/disable grad
        if semiotic_sgd:
            if semiotic_push:
                set_train([sender_percept])
                optim.semiosis_classifier(sender_percept, signal_game.sender, signal_game.receiver, log=log.debug)
            else:
                set_train([sender_percept, signal_game.sender, signal_game.receiver])
                optim.semiosis_joint(sender_percept, signal_game.sender, signal_game.receiver, log=log.debug)
        else:
            set_train([signal_game.sender, signal_game.receiver])
            optim.agents_only(sender_percept, signal_game.sender, signal_game.receiver, log=log.debug)

        log_whole_system_params(sender_percept, recv_percept, signal_game, log=log.debug)  
        
        if not semiotic_push:
            # a regular epoch
            epoch_log = train_fn(state, epoch, device, train_loader, 
                                 sender_percept, recv_percept, signal_game, _optim, approach, log)
            push_str = 'nopush'
        else:
            epoch_log = EpochHistory(epoch)
            if approach == 'proto':
                set_eval([sender_percept])
                
                # project back to true convs and tune signs classifier for this epoch
                # log nopush accuracies before
                weight_matrix_filename = 'outputL_weights'
                prototype_img_filename_prefix = 'prototype-img'
                prototype_self_act_filename_prefix = 'prototype-self-act'
                proto_bound_boxes_filename_prefix = 'bb'
    
                pack = tnt.test(model=sender_percept.model_multi, dataloader=ll_test_loader,
                                class_specific=True, log=log.info)
                epoch_log.log_concept_accuracies({'nopush': pack[2]})
                epoch_log.log_concept_costs({'nopush': pack})
                log.info(f"nopush sign concept test accuracy @ Epoch {epoch}:\t{pack[2]:.2f}")        
                util.save_enc_model(model=sender_percept.model, 
                                    model_dir=state['save_dir'], 
                                    model_name=str(epoch) + 'nopush', log=log.debug)

                push.push_prototypes(
                    train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel=sender_percept.model_multi, # pytorch network with prototype_vectors
                    class_specific=state['class_specific'],
                    preprocess_input_function=preprocess_input_function, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log.info
                )
                train_push_loader.reset()
                
                set_train([sender_percept])
                optim.semiosis_classifier(sender_percept, signal_game.sender, signal_game.receiver, log=log.debug)
                # convex optimization of classifier
                log.info(f"Start push sign concept convex optimization with {state['concept_classifier_epochs']} epochs.")
                concept_histories = []
                for c_epoch in range(state['concept_classifier_epochs']):
                    concept_history_path = os.path.join(state['save_dir'], f'{epoch}_push_cls_history.pkl')
                    concept_epoch_log = train.classifier_epoch(state, c_epoch, device, ll_train_loader, 
                                                               sender_percept, classifier_optimizer, log)
                    concept_histories.append(concept_epoch_log)
                    pickle_write(concept_history_path, concept_histories)
            else:
                # update classifier (CnnB and CW)
                log.info(f"Start convex optimization with {state['concept_classifier_epochs']} epochs.")
                concept_histories = []
                for c_epoch in range(state['concept_classifier_epochs']):
                    concept_history_path = os.path.join(state['save_dir'], f'{epoch}_push_cls_history.pkl')
                    concept_epoch_log = train.classifier_epoch(state, c_epoch, device, ll_train_loader, 
                                                               sender_percept, classifier_optimizer, log)
                    concept_histories.append(concept_epoch_log)
                    pickle_write(concept_history_path, concept_histories)

            push_str = 'push'


        # log agents task success
        metrics = semiotic_signal_test(sender_percept, signal_game, 
                                       sender_percept, recv_percept, 
                                       test_loader, device, return_metrics=True)
        epoch_log.log_accuracy(metrics['accuracy'])
        log.info(f"Receiver test accuracy @ Epoch {epoch}:\t{metrics['accuracy']:.2f}")

        # log signs model accuracies (semiotic prototype model only)
        if semiotic_sgd or semiotic_push:
            # joint_lr_scheduler.step()

            if approach == 'proto':
                pack = tnt.test(model=sender_percept.model_multi, dataloader=ll_test_loader,
                                class_specific=True, log=log.info)
                epoch_log.log_concept_accuracies({push_str: pack[2]})
                epoch_log.log_concept_costs({push_str: pack})
                log.info(f"{push_str} sign concept test accuracy @ Epoch {epoch}:\t{pack[2]:.2f}") 
            else:
                accu = ct.test(state, sender_percept.model_multi, ll_test_loader, log.info)
                epoch_log.log_concept_accuracies({push_str: accu})  

            util.save_enc_model(model=sender_percept.model, 
                                model_dir=state['save_dir'], 
                                model_name=str(epoch) + push_str, log=log.debug)

        train_loader.end_epoch(epoch_mode)
        test_loader.end_epoch(epoch_mode)

        epoch_histories.append(epoch_log)

        if epoch % state['checkpoint_interval'] == 0:
            torch.save(signal_game.sender.state_dict(), os.path.join(state['save_dir'], f'sender_e{epoch}.pth'))
            torch.save(signal_game.receiver.state_dict(), os.path.join(state['save_dir'], f'receiver_e{epoch}.pth'))
            log.info(f"Checkpointed at {state['save_dir']}")

        pickle_write(history_path, epoch_histories)

        del epoch_log  # failsafe

    return epoch_histories
