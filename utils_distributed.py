
import torch
import torch.distributed as dist
import numpy as np
import sys

def print_some(thing):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('print_output.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(thing)
        sys.stdout = original_stdout  # Reset the standard output to its original value

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


class Chunker:
    # The chunker creates a buffer we can use to store either the network params
    # or gradients to send a single message rather than having to send each param tensor
    # individually. This makes code simpler, especially for async, as you only need
    # to keep one request object for the buffer rather than a list for all param tensors
    # Also may improve speed as there is less overhead for the communication.
    def __init__(self,net,world_size,isMaster):
        ln = 0
        # work out size of the buffer
        for param in net.parameters():
            ln += np.prod(param.shape)
        self.buffer = torch.empty(ln,requires_grad=False).cuda()

        self.net = net

        if isMaster:
            self.reqs = [None for i in range(world_size-1)]
            self.grad_buffers = [self.buffer.clone() for i in range(world_size-1)]



    def grad_isend(self):
        # worker func
        self.flatten()
        self.gradreq = dist.isend(self.buffer, 0)


    def grad_irecv(self,workeri):
        # master func
        self.reqs[workeri-1]=dist.irecv(self.grad_buffers[workeri-1], workeri)


    def grad_wait(self):
        # worker func
        self.gradreq.wait()
        self.expand()

    def param_recv(self):
        # worker func
        dist.recv(self.buffer, 0)
        self.expand("data")

    def param_send(self,workeri):
        # master func
        self.flatten("data")
        dist.send(self.buffer,workeri)
        self.expand("data")

    def grad_ready(self,workeri):
        # master func
        return self.reqs[workeri].is_completed()

    def copy_grad(self,workeri):
        # master func
        # copy selected grad buffer back to the param tensors.
        i = 0
        for param in self.net.parameters():
            ln = np.prod(param.shape)
            x = self.grad_buffers[workeri].data[i:i + ln]

            param.grad.data = torch.reshape(x, param.size()).data
            i += ln



    def flatten(self,gradOrData="grad"):
        # flatten each param tensor and copy to buffer
        i = 0
        for param in self.net.parameters():
            if gradOrData=="grad":
                x = torch.flatten(param.grad.data)
            else:
                x = torch.flatten(param.data)
            self.buffer.data[i:i + len(x)] = x.data
            i += len(x)

    def expand(self,gradOrData="grad"):
        # copy buffer back to the param tensors.
        i = 0
        for param in self.net.parameters():
            ln = np.prod(param.shape)
            x = self.buffer.data[i:i + ln]
            if gradOrData=="grad":
                param.grad.data = torch.reshape(x, param.size()).data
            else:
                param.data = torch.reshape(x, param.size()).data
            i += ln


    def av_grad(self,world_size):

        self.flatten()
        # all-reduce once for the buffer
        dist.all_reduce(self.buffer.data, op=dist.ReduceOp.SUM)
        self.buffer.data /= world_size
        self.expand()




def av_loss(total_loss, n_samples):
    reduction = torch.FloatTensor([total_loss,n_samples]).cuda()
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    return float(reduction[0].item() / reduction[1].item())
