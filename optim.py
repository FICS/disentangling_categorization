import torch
import logging
log = logging.getLogger('optimizers')


def set_eval(models):
    for m in models:
        m.eval()
        

def set_train(models):
    for m in models:
        m.train()

        
def count_params(model):
    return sum([p.data.nelement() for p in model.parameters()])


def disable_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def enable_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True


def log_whole_system_params(sender_percept, recv_percept, _game, log=print):
    log('Optimized parameters:')
    log('sender_percept')
    for name, param in sender_percept.named_parameters():
        if param.requires_grad:
            log(f"\t{name}")

    log('sender')
    for name, param in _game.sender.named_parameters():
        if param.requires_grad:
            log(f"\t{name}")

    log('recv_percept')
    for name, param in recv_percept.named_parameters():
        if param.requires_grad:
            log(f"\t{name}")

    log('receiver')
    for name, param in _game.receiver.named_parameters():
        if param.requires_grad:
            log(f"\t{name}")


def agents_only(sender_percept, sender, receiver, log=print):
    sender_percept.model_multi.eval()
    sender.train()
    receiver.train()
    
    sender_percept.choose_grad('off', log=log)
    sender.choose_grad('on')
    receiver.choose_grad('on')
    
    log('\tSystem configuration: agents only')
    

def semiosis_joint(sender_percept, sender, receiver, log=print):
    sender_percept.model_multi.train()
    sender.train()
    receiver.train()
    
    sender_percept.choose_grad('joint', log=log)
    sender.choose_grad('on')
    receiver.choose_grad('on')
    
    log('\tSystem configuration: semiosis')
    
    
def semiosis_classifier(sender_percept, sender, receiver, log=print):
    sender_percept.model_multi.train() # only optimize fc
    sender.eval()
    receiver.eval()
    
    sender_percept.choose_grad('last_only', log=log)
    sender.choose_grad('off')
    receiver.choose_grad('off')
    
    log('\tSystem configuration: last layer optim')


def semiotic_social_optimizers(state, sender_percept, sender,
                               receiver_percept, receiver):
    # optimizers define
    # ==================================
    # simulate agents condition
    agents_only(sender_percept, sender, receiver, log=log.debug)

    sender_params_to_update = []
    for name,param in sender.named_parameters():
        if param.requires_grad == True:
            sender_params_to_update.append(param)

    recv_params_to_update = []
    for name,param in receiver.named_parameters():
        if param.requires_grad == True:
            recv_params_to_update.append(param)

    static_optimizer = torch.optim.Adam([
        {'params': sender_params_to_update, 
         'lr': state['sender_lr']},
        {'params': recv_params_to_update, 
         'lr': state['receiver_lr']}
    ])

    if len(state['semiotic_sgd_epochs']):
        for model in [sender_percept,  sender, receiver, receiver]:
            disable_parameter_requires_grad(model)
        
        # simulate semiosis condition
        semiosis_joint(sender_percept, sender, receiver, log=log.debug)
        
        sender_params_to_update = []
        for name,param in sender.named_parameters():
            if param.requires_grad == True:
                sender_params_to_update.append(param)

        recv_params_to_update = []
        for name,param in receiver.named_parameters():
            if param.requires_grad == True:
                recv_params_to_update.append(param)
            
        semiotic_optimizer_specs = \
        [
            {'params': sender_params_to_update, 
             'lr': state['sender_lr']},
            {'params': recv_params_to_update, 
             'lr': state['receiver_lr']}
        ]
        if state['approach'] == 'proto':
            semiotic_optimizer_specs.extend([
                {'params': sender_percept.model.features.parameters(), 
                 'lr': state['features_lr'], 
                 'weight_decay': 1e-3}, 
                {'params': sender_percept.model.add_on_layers.parameters(), 
                 'lr': state['add_on_layers_lr'], 
                 'weight_decay': 1e-3},
                {'params': sender_percept.model.prototype_vectors, 
                 'lr': state['prototype_vectors_lr']}
            ])
            classifier_optimizer_specs = [
                {'params': sender_percept.model.last_layer.parameters(), 
                 'lr': state['last_layer_lr']}
            ]
        else:
            semiotic_optimizer_specs.append(
                {'params': sender_percept.model.base_model.parameters(), 
                 'lr': state['features_lr'], 
                 'weight_decay': 1e-3}, 
            )
            classifier_optimizer_specs = [
                {'params': sender_percept.model.classifier.parameters(), 
                 'lr': state['last_layer_lr']}
            ]
            
        classifier_optimizer = torch.optim.Adam(classifier_optimizer_specs)
        semiotic_optimizer = torch.optim.Adam(semiotic_optimizer_specs)
        # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(semiotic_optimizer, step_size=10, gamma=0.1)

        return static_optimizer, semiotic_optimizer, classifier_optimizer
    else:
        return static_optimizer, None, None
