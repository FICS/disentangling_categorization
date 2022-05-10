import torch
import torch.nn as nn
import numpy as np


class SignalGameGS(nn.Module):
    def __init__(self, sender, receiver, loss, length_cost=0.0, sys_losses=[], aux_losses=[]):
        super(SignalGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.length_cost = length_cost
        self.aux_losses = aux_losses
        self.sys_losses = sys_losses
    
    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output, receiver_hiddens = self.receiver(message, receiver_input)
        
        loss = 0.
        aux_loss_memory = []
        
        # keep track of which sample have been eos already
        # P(no eos happened before| step)
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0
        
        z = 0.0
        aux_info = {}
        # get the expected loss as a sum of over all steps upto max_len
        # can be 0 until eos is reached for one-hot case (expected loss vs. actual loss)
        for step in range(receiver_output.size(1)):
            
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
            )
                
            eos_mask = message[:, step, 0]  # always eos == 0
            
            # P(eos | step) * P(no eos happened before| step)
            add_mask = eos_mask * not_eosed_before
            z += add_mask
            # if 1. in add_mask:
                # print('step:', step, '\nmask:', add_mask, '\nmessage:', message.argmax(dim=-1))
            
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)
            
            # save numerical aux information from main task loss and add
            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)
            
            # Auxillary losses
            for aux_loss_fn, weighting in self.aux_losses:
                aux_loss, step_aux = aux_loss_fn(
                    sender_input,
                    step,
                    message,
                    receiver_input,
                    receiver_output,
                    labels,
                    self.sender.vocab_size
                )
                weighted_aux = weighting * aux_loss * add_mask
                # loss += weighted_aux
                aux_loss_memory.append(weighted_aux)
                
                # save numerical aux information from aux loss and add
                aux_name = aux_loss_fn.__name__
                aux_info[aux_name] = weighted_aux + aux_info.get(aux_name, 0.0)
                
                for name, value in step_aux.items():
                    aux_info[name] = value * add_mask + aux_info.get(name, 0.0)
            
            for weighted_aux_loss in aux_loss_memory:
                loss += weighted_aux_loss * not_eosed_before
            
            # update mask:
            # P(no eos happened before| step)
            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        # Aux losses
        for weighted_aux_loss in aux_loss_memory:
            loss += weighted_aux_loss * not_eosed_before
        
        # Reduce to apply system losses
        loss = loss.mean()
        
        # System losses
        for sys_loss_fn, weighting in self.sys_losses:
            sys_loss, _aux = sys_loss_fn(
                sender_input,
                message,
                receiver_input,
                receiver_output,
                labels,
                self.sender.vocab_size
            )
            weighted_sys = weighting * sys_loss
            aux_info[sys_loss_fn.__name__] = np.mean(weighted_sys.detach().cpu().numpy())
            loss += weighted_sys
                
        expected_length += (step + 1) * not_eosed_before
        aux_info['expected_length'] = torch.mean(expected_length.detach().cpu()).numpy()
        
        # make sure eos probability over length of the sequence sums to 1
        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"
        
        # aux_info['sender_message'] = message.detach()
        # aux_info['receiver_output'] = receiver_output.detach()
        
        # average over minibatch
        for aux_loss_fn, _ in self.aux_losses:
            aux_loss_name = aux_loss_fn.__name__
            aux_info[aux_loss_name] = np.mean(aux_info[aux_loss_name].detach().cpu().numpy())
        
        return loss, message, receiver_output, aux_info


class ReificationAEGameGS(SignalGameGS):
    def __init__(self, sender, receiver, loss, length_cost=0, sys_losses=[], aux_losses=[]):
        super().__init__(sender, receiver, loss, length_cost, sys_losses, aux_losses)
    
    def forward(self, sender_input, labels, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output, receiver_hiddens = self.receiver(message, receiver_input)
        
        loss = 0.
        aux_loss_memory = []
        
        # keep track of which sample have been eos already
        # P(no eos happened before| step)
        not_eosed_before = torch.ones(receiver_output.size(0)).to(
            receiver_output.device
        )
        expected_length = 0.0
        
        z = 0.0
        aux_info = {}
        # get the expected loss as a sum of over all steps upto max_len
        # can be 0 until eos is reached for one-hot case (expected loss vs. actual loss)
        for step in range(receiver_output.size(1)):
            step_loss, step_aux = self.loss(
                sender_input,
                message[:, step, ...],
                receiver_input,
                receiver_output[:, step, ...],
                labels,
            )
                
            eos_mask = message[:, step, 0]  # always eos == 0
            
            # P(eos | step) * P(no eos happened before| step)
            add_mask = eos_mask * not_eosed_before
            z += add_mask
            # if 1. in add_mask:
                # print('step:', step, '\nmask:', add_mask, '\nmessage:', message.argmax(dim=-1))
            
            loss += step_loss * add_mask + self.length_cost * (1.0 + step) * add_mask
            expected_length += add_mask.detach() * (1.0 + step)
            
            # save numerical aux information from main task loss and add
            for name, value in step_aux.items():
                aux_info[name] = value * add_mask + aux_info.get(name, 0.0)
            
            # Auxillary losses
            for aux_loss_fn, weighting in self.aux_losses:
                if aux_loss_fn.__name__ == 'Lp_reconstruction_loss':
                    aux_loss, step_aux = aux_loss_fn(
                        self.sender.recons[:, step, ...],
                        self.receiver.recons[:, step, ...],
                    )
                else:
                    aux_loss, step_aux = aux_loss_fn(
                        sender_input,
                        step,
                        message,
                        receiver_input,
                        receiver_output,
                        labels,
                        self.sender.vocab_size
                    )
                    
                weighted_aux = weighting * aux_loss * add_mask
                # loss += weighted_aux
                aux_loss_memory.append(weighted_aux)
                
                # save numerical aux information from aux loss and add
                aux_name = aux_loss_fn.__name__
                aux_info[aux_name] = weighted_aux + aux_info.get(aux_name, 0.0)
                
                for name, value in step_aux.items():
                    aux_info[name] = value * add_mask + aux_info.get(name, 0.0)
            
            for weighted_aux_loss in aux_loss_memory:
                loss += weighted_aux_loss * not_eosed_before
            
            # update mask:
            # P(no eos happened before| step)
            not_eosed_before = not_eosed_before * (1.0 - eos_mask)

        # the remainder of the probability mass
        loss += (
            step_loss * not_eosed_before
            + self.length_cost * (step + 1.0) * not_eosed_before
        )
        # Aux losses
        for weighted_aux_loss in aux_loss_memory:
            loss += weighted_aux_loss * not_eosed_before
        
        # Reduce to apply system losses
        loss = loss.mean()
        
        # System losses
        for sys_loss_fn, weighting in self.sys_losses:
            sys_loss, _aux = sys_loss_fn(
                sender_input,
                message,
                receiver_input,
                receiver_output,
                labels,
                self.sender.vocab_size
            )
            weighted_sys = weighting * sys_loss
            aux_info[sys_loss_fn.__name__] = np.mean(weighted_sys.detach().cpu().numpy())
            loss += weighted_sys
                
        expected_length += (step + 1) * not_eosed_before
        aux_info['expected_length'] = torch.mean(expected_length.detach().cpu()).numpy()
        
        # make sure eos probability over length of the sequence sums to 1
        z += not_eosed_before
        assert z.allclose(
            torch.ones_like(z)
        ), f"lost probability mass, {z.min()}, {z.max()}"
        
        # aux_info['sender_message'] = message.detach()
        # aux_info['receiver_output'] = receiver_output.detach()
        
        # average over minibatch
        for aux_loss_fn, _ in self.aux_losses:
            aux_loss_name = aux_loss_fn.__name__
            aux_info[aux_loss_name] = np.mean(aux_info[aux_loss_name].detach().cpu().numpy())
        
        return loss, message, receiver_output, aux_info
