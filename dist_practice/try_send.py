import torch
import sys
sys.path.append("../")
from distributed import init_workers
import torch.distributed as dist
import threading


distributed_backend = 'gloo'

global_rank, world_size = init_workers(distributed_backend)



print(f"my global rank is {global_rank}")
ranks_per_node = torch.cuda.device_count()
print(f"there are {ranks_per_node} devices on this node")

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

################################################################################
# Synchronous send/recv
################################################################################
i = 1.0
try:
    if global_rank==0:
        message = i*torch.tensor([[1., -1.], [1., -1.]])
        dist.send(message,1)

    if global_rank==1:
        message_recv = torch.zeros(2,2)
        dist.recv(message_recv,0)
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message_recv}")
except Exception as ex:
    print(ex)
    print(f"FAILED test {i}")


################################################################################
# Synchronous send/recv GPU-GPU
################################################################################
i += 1
try:
    if global_rank==0:
        message = i*torch.tensor([[1., -1.], [1., -1.]]).cuda()
        if distributed_backend=="gloo":
            message=message.cpu()
        dist.send(message,1)

    if global_rank==1:
        message_recv = torch.zeros(2,2).cuda()
        if distributed_backend=="gloo":
            message_recv = message_recv.cpu()
        dist.recv(message_recv,0)
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message_recv}")
except Exception as ex:
    print(ex)
    print(f"FAILED test {i}")


################################################################################
# Synchronous broadcast
################################################################################
i += 1
try:
    if global_rank==0:
        message = i * torch.tensor([[1., -1.], [1., -1.]])
    else:
        message = torch.zeros(2, 2)

    dist.broadcast(message,0)
    print(f"\n\n\n\n\nrank {global_rank}: I have received {message}")
except Exception as ex:
    print(ex)
    print(f"FAILED test {i}")


################################################################################
# Synchronous broadcast from GPU
################################################################################
if (dist.get_world_size()==2) and torch.cuda.is_available():
    i += 1
    try:
        if global_rank==0:
            message = i * torch.tensor([[1., -1.], [1., -1.]])
            message = message.cuda()
        else:
            message = torch.zeros(2, 2)
            message = message.cuda()

        dist.broadcast(message,0)
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message}")
    except Exception as ex:
        print(ex)
        print(f"FAILED test {i}")

################################################################################
# asynchronous send/recv with req.wait()
################################################################################
i += 1.0
try:
    if global_rank==0:
        message = i*torch.tensor([[1., -1.], [1., -1.]])
        req = dist.isend(message,1)
        # useful stuff here
        req.wait()

    if global_rank==1:
        message_recv = torch.zeros(2,2)
        req = dist.irecv(message_recv,0)
        #useful stuff here
        req.wait()


        print(f"\n\n\n\n\nrank {global_rank}: I have received {message_recv}.")
except Exception as ex:
    print(f"FAILED test {i}")
    print(ex)


################################################################################
# asynchronous send/recv with workaround for is_completed() bug. See
# https://github.com/pytorch/pytorch/issues/30723
################################################################################
def daemon_thread(req):
    req.wait()

i += 1.0
try:
    if global_rank==0:
        message = i*torch.tensor([[1., -1.], [1., -1.]])
        req = dist.isend(message,1)
        # useful stuff here
        req.wait()

    if global_rank==1:
        message_recv = torch.zeros(2,2)
        req = dist.irecv(message_recv,0)
        t = threading.Thread(target=daemon_thread, args=(req,), daemon=True)
        something = 0
        t.start()
        while t.is_alive():
            something += 1


        print(something)

        print(f"\n\n\n\n\nrank {global_rank}: I have received {message_recv}.")
except Exception as ex:
    print(f"FAILED test {i}")
    print(ex)


################################################################################
# asynchronous send/recv with req.is_completed() (MPI only, gloo will hang as of
# Pytorch 1.8.
################################################################################
i += 1.0
if distributed_backend=="mpi":
    try:
        if global_rank==0:
            message = i*torch.tensor([[1., -1.], [1., -1.]])
            req = dist.isend(message,1)
            # useful stuff here
            req.wait()

        if global_rank==1:
            message_recv = torch.zeros(2,2)
            req = dist.irecv(message_recv,0)
            something = 0

            while not req.is_completed():
                something += 1


            print(something)

            print(f"\n\n\n\n\nrank {global_rank}: I have received {message_recv}.")
    except Exception as ex:
        print(f"FAILED test {i}")
        print(ex)


################################################################################
# Send/recv with groups and broadcasts. Hack to allow to use NCCL for M/S
# environment
################################################################################
if dist.get_world_size()>=3:
    i+=1.0
    # all ranks must envoke new group even if not in the new group
    g01 = dist.new_group([0,1])
    g02 = dist.new_group([0,2])
    if global_rank==0:
        message1 = i * torch.tensor([[1., -1.], [1., -1.]])
        i+=1.0
        message2 = i * torch.tensor([[1., -1.], [1., -1.]])
        dist.broadcast(message1,0, g01)
        dist.broadcast(message2, 0, g02)

    if global_rank==1:
        message1 = torch.zeros(2,2)
        dist.broadcast(message1, 0, g01)
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message1}.")

    if global_rank==2:
        message2 = torch.zeros(2,2)
        dist.broadcast(message2, 0, g02)
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message2}.")


################################################################################
# Async Send/recv with groups and broadcasts. Hack to allow to use NCCL for M/S
# environment
################################################################################
if dist.get_world_size()>=3:
    i+=1.0
    g01 = dist.new_group([0,1])
    g02 = dist.new_group([0,2])
    if global_rank==0:
        message1 = i * torch.tensor([[1., -1.], [1., -1.]])
        i+=1.0
        message2 = i * torch.tensor([[1., -1.], [1., -1.]])
        req1 = dist.broadcast(message1,0, g01,async_op=True)
        req2 = dist.broadcast(message2, 0, g02,async_op=True)
        req1.wait()
        req2.wait()

    if global_rank==1:
        message1 = torch.zeros(2,2)
        req = dist.broadcast(message1, 0, g01,async_op=True)
        req.wait()
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message1}.")

    if global_rank==2:
        message2 = torch.zeros(2,2)
        req = dist.broadcast(message2, 0, g02,async_op=True)
        req.wait()
        print(f"\n\n\n\n\nrank {global_rank}: I have received {message2}.")


