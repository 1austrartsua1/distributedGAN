
import torch
import torch.distributed as dist
import numpy as np

def av_param(model,world_size):

    # synchronize the models on all the different devices to be the average
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size







def av_grad(model, world_size):

    #for param in model.parameters():
    for name,param in model.named_parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
        else:
            print("Got a gradient which is None:")
            print(name)








class GradAverager:
    def __init__(self,net):
        ln = 0
        # work out size of the buffer
        for param in net.parameters():
            ln += np.prod(param.shape)
        self.buffer = torch.empty(ln,requires_grad=False).cuda()

    def av_grad(self,net, world_size):
        #flatten each param tensor and copy to buffer
        i = 0
        for param in net.parameters():
            x = torch.flatten(param.grad.data)
            self.buffer.data[i:i+len(x)] = x.data
            i += len(x)

        # all-reduce once for the buffer
        dist.all_reduce(self.buffer.data, op=dist.ReduceOp.SUM)
        self.buffer.data /= world_size

        # copy buffer back to the param tensors.
        i = 0
        for param in net.parameters():
            ln = np.prod(param.shape)
            x = self.buffer.data[i:i+ln]
            param.grad.data = torch.reshape(x,param.size()).data
            i += ln



def av_loss(total_loss, n_samples):
    reduction = torch.FloatTensor([total_loss,n_samples]).cuda()
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    return float(reduction[0].item() / reduction[1].item())
