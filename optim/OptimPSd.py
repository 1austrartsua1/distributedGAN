#  MIT License

# Copyright (c) Facebook, Inc. and its affiliates.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# written by Hugo Berard (berard.hugo@gmail.com) while at Facebook.


import math
import torch
from torch.optim import Optimizer
import torch.distributed as dist



class PS(Optimizer):
    """Base class for optimizers with extrapolation step.

        Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    def __init__(self, params, defaults):
        super(PS, self).__init__(params, defaults)
        self.params_copy = []
        self.wi = []



    def update(self, p, group, update_type):
        raise NotImplementedError

    def extrapolate(self):
        """Performs the extrapolation step and save a copy of the current parameters for the update step.
        """
        # Check if a copy of the parameters was already made.
        is_empty = (len(self.params_copy) == 0)
        no_dual = (len(self.wi) == 0)
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if no_dual:
                    self.wi.append(torch.zeros_like(p))
                u = self.update(p, group, "extrap", self.wi[i])
                i += 1

                if is_empty:
                    # Save the current parameters for the update step.
                    # Several extrapolation step can be made before each update
                    # but only the parameters before the first extrapolation step are saved.
                    self.params_copy.append(p.data.clone())
                if u is None:
                    continue
                # Update the current parameters
                p.data.add_(u)

    def primal_step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if len(self.params_copy) == 0:
            raise RuntimeError('Need to call extrapolation before calling step.')

        loss = None
        if closure is not None:
            loss = closure()

        i = -1
        for group in self.param_groups:
            for p in group['params']:
                i += 1
                u = self.update(p, group, "step")
                if u is None:
                    continue
                # Update the parameters saved during the extrapolation step
                temp = p.data.clone()
                p.data = self.params_copy[i].add_(u)
                self.params_copy[i].data = temp #params_copy now holds x_i

        return loss

    def dual_step(self,world_size):
        #must be called after primal_step
        # first average x_i to get xbar
        params_av = []
        for i in range(len(self.params_copy)):
            params_av.append(self.params_copy[i].data.clone())
            dist.all_reduce(params_av[-1].data, op=dist.ReduceOp.SUM)
            params_av[-1].data /= world_size

        self.update_duals(params_av)

        self.params_copy = []
        params_av = []

    def update_duals(self,params_av):
        for i in range(len(params_av)):
            self.wi[i].data -= self.lr_dual*(self.params_copy[i].data - params_av[i].data)
        return




class PS_SGD(PS):
    """Implements stochastic gradient descent with extrapolation step (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr_step (float): learning rate for the .step() update
        lr_extrap (float): learning rate for the .extrapolate() update (by default equal to lr_step)
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.ExtraSGD(model.parameters(), lr_step=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.extrapolation()
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    def __init__(self, params, lr_step, lr_extrap=None, lr_dual=None):
        if lr_step < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr_step))

        if lr_extrap is not None and lr_step < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr_extrap))


        if lr_extrap is None:
            lr_extrap = lr_step
        if lr_dual is None:
            self.lr_dual = lr_step
        else:
            self.lr_dual = lr_dual

        defaults = dict(lr_step=lr_step, lr_extrap=lr_extrap)

        super(PS_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PS_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update(self, p, group, update_type, dual=None):

        if p.grad is None:
            return None
        if update_type == "extrap":
            d_p = p.grad.data - dual.data
        else:
            d_p = p.grad.data

        return -group['lr_'+update_type]*d_p

class PS_Adam(PS):
    """Implements the Adam algorithm with extrapolation step.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr_step (float, optional): learning rate for the .step() update (default: 1e-3)
        lr_extrap (float, optional):learning rate for the .extrap() update (default: lr_step)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    """

    def __init__(self, params, lr_step=1e-3, lr_extrap=None, betas=(0.9, 0.999), eps=1e-8,
                 amsgrad=False,lr_dual=None,AdamForDuals = False):
        if not 0.0 <= lr_step:
            raise ValueError("Invalid learning rate: {}".format(lr_step))
        if (lr_extrap is not None) and (not 0.0 <= lr_extrap):
            raise ValueError("Invalid learning rate: {}".format(lr_extrap))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if lr_extrap is None:
            lr_extrap = lr_step
        if lr_dual is None:
            self.lr_dual = lr_step
        else:
            self.lr_dual = lr_dual



        #params needed for dual with adam updates
        self.AdamForDuals = AdamForDuals
        self.dualAdamInitializeNeeded = True
        self.dual_beta1 = betas[0]
        self.dual_beta2 = betas[1]
        self.dual_exp_avg = []
        self.dual_exp_avg_sq = []
        self.dual_eps = eps
        self.dualstep = 1

        defaults = dict(lr_step=lr_step,lr_extrap=lr_extrap, betas=betas, eps=eps,
                        amsgrad=amsgrad)

        super(PS_Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(PS_Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def update(self, p, group, update_type, dual=None):
        if p.grad is None:
            return None
        if update_type == "extrap":
            grad = p.grad.data - dual.data
        else:
            grad = p.grad

        if grad.is_sparse:
            raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
        amsgrad = group['amsgrad']

        state = self.state[p]

        # State initialization
        if state.get('initialized_'+update_type) is None:
            state['step'+update_type] = 0
            # Exponential moving average of gradient values
            state['exp_avg'+update_type] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'+update_type] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'+update_type] = torch.zeros_like(p.data)
            state['initialized_'+update_type] = 1

        exp_avg, exp_avg_sq = state['exp_avg'+update_type], state['exp_avg_sq'+update_type]
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq'+update_type]
        beta1, beta2 = group['betas']

        state['step'+update_type] += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step'+update_type]
        bias_correction2 = 1 - beta2 ** state['step'+update_type]
        step_size = group['lr_'+update_type] * math.sqrt(bias_correction2) / bias_correction1

        return -step_size*exp_avg/denom

    def update_duals(self,params_av):

        if not self.AdamForDuals:
            for i in range(len(params_av)):
                self.wi[i].data -= self.lr_dual*(self.params_copy[i].data - params_av[i].data)
            return

        beta1 = self.dual_beta1
        beta2 = self.dual_beta2
        for i in range(len(params_av)):
            update = self.params_copy[i].data - params_av[i].data
            if self.dualAdamInitializeNeeded:
                self.dual_exp_avg.append(torch.zeros_like(params_av[i].data))
                self.dual_exp_avg_sq.append(torch.zeros_like(params_av[i].data))


            self.dual_exp_avg[i].mul_(beta1).add_(1 - beta1, update)
            self.dual_exp_avg_sq[i].mul_(beta2).addcmul_(1 - beta2, update, update)
            denom = self.dual_exp_avg_sq[i].sqrt().add_(self.dual_eps)
            bias_correction1 = 1 - beta1 ** self.dualstep
            bias_correction2 = 1 - beta2 ** self.dualstep
            step_size = self.lr_dual * math.sqrt(bias_correction2) / bias_correction1

            self.wi[i].data -= step_size*self.dual_exp_avg[i]/denom

        self.dualAdamInitializeNeeded = False
        self.dualstep += 1
