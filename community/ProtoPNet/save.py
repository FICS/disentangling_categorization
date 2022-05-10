import os
import torch

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
#         torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def save_best(model, model_dir, accu, epoch, log=print):
    '''
    model: this is not the multigpu model
    '''
    log(f'\tBest model so far: epoch={epoch}, accu={accu:.3f}')
    # save special format for later 
    model_state = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'accu': accu,
        'prototype_shape': model.prototype_shape,
    }
    torch.save(obj=model_state, f=os.path.join(model_dir, 'best.pth'))
#         torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
