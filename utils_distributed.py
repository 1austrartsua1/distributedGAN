
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
    def __init__(self,net,world_size,isMaster,backend="mpi",groups = None, global_rank = -1):
        ln = 0
        # work out size of the buffer
        for param in net.parameters():
            ln += np.prod(param.shape)
        self.buffer = torch.empty(ln,requires_grad=False).cuda()

        self.net = net

        if isMaster:
            self.reqs = [None for i in range(world_size-1)]
            self.grad_buffers = [self.buffer.data.clone() for i in range(world_size-1)]

        else:
            self.gradbuffer = torch.empty(ln, requires_grad=False).cuda()

        self.backend = backend
        self.groups = groups
        self.global_rank = global_rank


    def isend_wrapper(self,tensor):
        if self.backend == "nccl":
            return dist.broadcast(tensor, self.global_rank, self.groups[self.global_rank], async_op=True)
        else:
            return dist.isend(tensor, 0)

    def irecv_wrapper(self,tensor, sending_node):
        if self.backend == "nccl":
            return dist.broadcast(tensor, sending_node, self.groups[sending_node], async_op=True)
        else:
            return dist.irecv(tensor, sending_node)


    def send_wrapper(self,tensor, node):
        if self.backend == "nccl":
            dist.broadcast(tensor, 0, self.groups[node])
        else:
            dist.send(tensor, node)

    def recv_wrapper(self,tensor):
        if self.backend == "nccl":
            dist.broadcast(tensor, 0, self.groups[self.global_rank])
        else:
            dist.recv(tensor, 0)



    def getBufferNorm(self,which="grad"):
        if which == "grad":
            return torch.norm(self.gradbuffer) ** 2
        else:
            return torch.norm(self.buffer)**2

    def getGradBufferNorm(self,workeri):
        return torch.norm(self.grad_buffers[workeri-1].data)**2

    def grad_isend(self):
        # worker func
        self.workerflatten()
        norms = self.getBufferNorm()
        #self.gradreq = dist.isend(self.gradbuffer.data, 0)
        self.gradreq = self.isend_wrapper(self.gradbuffer.data)
        return norms


    def grad_irecv(self,workeri):
        # master func
        #self.reqs[workeri-1]=dist.irecv(self.grad_buffers[workeri-1].data, workeri)
        self.reqs[workeri-1] = self.irecv_wrapper(self.grad_buffers[workeri-1].data, workeri)


    def grad_wait(self):
        # worker func
        self.gradreq.wait()
        #self.expand()

    def param_recv(self):
        # worker func
        #dist.recv(self.buffer.data, 0)
        self.recv_wrapper(self.buffer.data)
        self.expand("data")

    def param_send(self,workeri):
        # master func
        self.flatten("data")
        #dist.send(self.buffer.data,workeri)
        self.send_wrapper(self.buffer.data, workeri)
        #self.expand("data")

    def grad_ready(self,workeri):
        # master func
        return self.reqs[workeri].is_completed()

    def copy_grad(self,workeri):
        # master func
        # copy selected grad buffer back to the param grad tensors.
        i = 0
        norms = 0.0
        for param in self.net.parameters():
            ln = np.prod(param.shape)
            x = self.grad_buffers[workeri].data[i:i + ln]
            norms += torch.norm(x)**2

            param.grad.data = torch.reshape(x, param.size()).data.clone()

            i += ln

        return norms

    def workerflatten(self):
        # flatten each param tensor and copy to buffer
        i = 0
        for param in self.net.parameters():

            x = torch.flatten(param.grad.data)

            self.gradbuffer.data[i:i + len(x)] = x.data.clone()
            i += len(x)

    def flatten(self,gradOrData="grad"):
        # flatten each param tensor and copy to buffer
        i = 0
        for param in self.net.parameters():
            if gradOrData=="grad":
                x = torch.flatten(param.grad.data)
            else:
                x = torch.flatten(param.data)
            self.buffer.data[i:i + len(x)] = x.data.clone()
            i += len(x)

    def expand(self,gradOrData="grad"):
        # copy buffer back to the param tensors.
        i = 0
        for param in self.net.parameters():
            ln = np.prod(param.shape)
            x = self.buffer.data[i:i + ln]
            if gradOrData=="grad":
                param.grad.data = torch.reshape(x, param.size()).data.clone()
            else:
                param.data = torch.reshape(x, param.size()).data.clone()
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
