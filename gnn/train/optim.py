import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class Frankenstein(Optimizer):
    r"""Implements Frankenstein optimizer
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        fixed_beta (float, optional):  when fixed_beta!=0, the beta 
            is performed as a constant value
            when when fixed_beta==0, the beta depend on learning rate
            automatically (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        base_lr (float, optional): The default learning rate paired with beta. If training from scratch, set it to 1e-3; for fine-tuning, set it to 1e-4. (Default: 1e-3)
        base_beta (float, optional): default beta coefficient (default: 0.9)
    """
    def __init__(self, params, lr=1e-3, eps=1e-8,
                 weight_decay=0, weight_decouple=True, fixed_beta=0,base_lr=1e-3,base_beta=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= fixed_beta < 1.0:
            raise ValueError("Invalid momentum value: {}".format(fixed_beta))
        defaults = dict(lr=torch.tensor(lr,dtype=torch.bfloat16), eps=torch.tensor(eps,dtype=torch.bfloat16),
                        weight_decay=torch.tensor(weight_decay,dtype=torch.bfloat16), weight_decouple=weight_decouple,
                        fixed_beta=fixed_beta,base_lr=base_lr,base_beta=base_beta
                        )
        super(Frankenstein, self).__init__(params, defaults)
        
        
        self.max_xi=float(np.exp(1.03))
        self.min_xi=float(np.exp(-0.2))
        
        self.max_beta_adj=float(0.05)
        self.min_beta_adj=float(1.0)
        
        
    def __setstate__(self, state):
        super(Frankenstein, self).__setstate__(state)
        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Frankenstein does not support sparse gradients, please consider SparseAdam instead')
                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['s'] = torch.mul(torch.ones_like(p, memory_format=torch.preserve_format),group['lr'])
                    state['vmax'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                m, s,vmax = state['m'], state['s'],state['vmax']
                
                
                if group['fixed_beta']!=0:
                    momentum=group['fixed_beta']
                else:
                    momentum=1.0-np.clip((1.0-group['base_beta'])*math.sqrt(group['lr']/group['base_lr']),self.max_beta_adj,self.min_beta_adj)
                
                if group['weight_decay'] > 0:
                    if group['weight_decouple']:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])
                p_factor=torch.div(torch.acos(torch.tanh(torch.mul(m,grad))),math.pi)                
                dfc =torch.div(1.60653065971,torch.add(1.0,torch.exp(-torch.abs(torch.add(s ,-p_factor)))))
                square_grad=torch.add(torch.mul(grad,grad) ,group['eps'])
                
                max_square_grad=torch.max(vmax, square_grad)
                max_grad=torch.sqrt(square_grad)
                
                lr_t=torch.mul(torch.div(group['lr'],max_grad),dfc)
                xi_factor=torch.log(torch.clamp(3.21828182846-p_factor+max_grad, min=self.min_xi,max=self.max_xi))
                m.mul_(torch.mul(xi_factor,momentum)).add_(torch.mul(-grad , lr_t))
                beta_2=torch.mul(torch.clamp(torch.div(square_grad,s),0.0,1.0),torch.abs(p_factor-0.5))
                p.data.add_(torch.add(torch.mul(momentum,m),torch.mul(-grad, lr_t)))
                vmax.copy_(torch.add(torch.mul(max_square_grad,torch.add(1.0,-beta_2)),torch.mul(beta_2,square_grad)))
                s.copy_(square_grad)
        return loss