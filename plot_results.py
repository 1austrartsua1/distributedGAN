

import pickle
import matplotlib.pyplot as plt


def print_a_res(file):
    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)

    print(res)

def calculate_scaling_result(file):
    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)


    # weak scaling results
    total_forward_steps = res['forwardStepStamps'][-1]
    worldsize = res['world_size']
    totalActiveRuntime = res['timestamps'][-1]
    forwardStepsPerSecondPerGPU = total_forward_steps/totalActiveRuntime


    print(f"algorithm={res['algorithm']}, worldsize={worldsize}")

    print(f"total_forward_steps = {total_forward_steps}")

    print(f"totalActiveRuntime={totalActiveRuntime}")
    print(f"forwardStepsPerSecondPerGPU={forwardStepsPerSecondPerGPU}")

def plot_loss(file,plotType,label):
    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)
    if plotType == "time":
        plt.plot(res['timestamps'],res['G_losses'],'o-',label="G_loss "+label)
        plt.plot(res['timestamps'],res['D_losses'],'o-',label="D_loss "+label)
        plt.ylabel('losses')
        plt.xlabel('time (s)')
    else:
        plt.plot(res['forwardStepStamps'],res['G_losses'],'o-',label="G_loss "+label)
        plt.plot(res['forwardStepStamps'],res['D_losses'],'o-',label="D_loss "+label)
        plt.ylabel('losses')
        plt.xlabel('num forward steps')



def get_a_plot(file,plotType,label):

    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)

    #print(f"total runtime = {res['total_running_time']/60:.4f} minutes")



    if plotType == "time":
        plt.plot(res['timestamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('time (s)')


    else:
        plt.plot(res['forwardStepStamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('num forward steps')

    return


if __name__ == "__main__":

    get_a_plot("extragrad/extragrad8","time","extragrad-batchsize=8*64")
    get_a_plot("fbf/fbf_distBigBatch1","time","fbf-batchsize=8*64")
    get_a_plot("fbf/fbf_replicate1","time","fbf-batchsize=1*64")
    
    file = "ps/ps1beta0"
    label = "ps1beta0"
    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)
    plt.plot(res['timestamps'][1:],res['iscores'],'o-',label=label)


    plt.grid()
    plt.legend()
    plt.show()
