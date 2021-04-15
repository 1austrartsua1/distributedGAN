

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

    print(f"total runtime = {res['total_running_time']/60:.4f} minutes")



    if plotType == "time":
        plt.plot(res['timestamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('time (s)')


    else:
        plt.plot(res['forwardStepStamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('num forward steps')

    return


get_a_plot("ps/ps1","forward","ps1")
get_a_plot("ps/ps2","forward","ps2")
get_a_plot("ps/ps4","forward","ps4")
get_a_plot("ps/ps8","forward","ps8")
plt.grid()
plt.legend()
plt.show()

if __name__ == "__main__2":

    get_a_plot("extragrad/extragrad8","time","extragrad-batchsize=8*64")
    get_a_plot("fbf/fbf_distBigBatch1","time","fbf-batchsize=8*64")
    get_a_plot("ps/ps8","time","ps-batchsize=8*64")
    get_a_plot("fbf/fbf_replicate1","time","fbf-batchsize=1*64")
    get_a_plot("ps/results_14_04_2021::07:12:54","time","ps-batchsize=1*64")
    #get_a_plot("ps/results_14_04_2021::06:59:29","time","ps-batchsize=2*64-clip2")
    get_a_plot("ps/ps4","time","ps-4")






    plt.grid()
    plt.legend()
    plt.show()

    get_a_plot("extragrad/extragrad8","forward","extragrad-batchsize=8*64")
    get_a_plot("fbf/fbf_distBigBatch1","forward","fbf-batchsize=8*64")
    get_a_plot("ps/ps8","forward","ps-batchsize=8*64")

    plt.grid()
    plt.legend()
    plt.show()
