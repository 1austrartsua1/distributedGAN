
import torch
import torch.distributed as dist

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





def av_loss(total_loss, n_samples):
    reduction = torch.FloatTensor([total_loss,n_samples]).cuda()
    dist.all_reduce(reduction, op=dist.ReduceOp.SUM)
    return float(reduction[0].item() / reduction[1].item())
