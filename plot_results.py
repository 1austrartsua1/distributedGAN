

import pickle
import matplotlib.pyplot as plt
import numpy as np



def print_a_res(file):
    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)

    for key in res.keys():
        if key != 'worker_order':
            print(f"{key}: {res[key]}")

    if res.get('worker_order'):
        dist = []
        currLen = 0
        for i in range(len(res['worker_order'])):
            if res['worker_order'][i]>=currLen:
                for _ in range(res['worker_order'][i]-currLen + 1):
                    dist.append(0)
                currLen = len(dist)

            dist[res['worker_order'][i]] += 1

        print(dist)
        mx = -float('inf')
        mn = float('inf')
        av = 0.0
        for d in dist:
            av += d
            if d > mx:
                mx = d

            if d < mn:
                mn = d
        av = av/len(dist)
        print(f"max relative discrepancy = {100*(mx-mn)/av:.3f}%")


        #plt.hist(res['worker_order'],bins=7)
        #plt.show()




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

    print(f"all reduce fraction of time on av: {res['reduceFracAv']}")

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



def get_a_plot(file,plotType,label,commsPerForwardStep=None):

    with open('results/'+file, 'rb') as handle:
        res = pickle.load(handle)

    #print(f"total runtime = {res['total_running_time']/60:.4f} minutes")



    if plotType == "time":
        plt.plot(res['timestamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('time (s)')

    elif plotType == "comms":
        comms = [commsPerForwardStep*res['forwardStepStamps'][i] for i in range(len(res['forwardStepStamps']))]
        plt.plot(comms,res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('num gen/dis all-reduces')
    else:
        plt.plot(res['forwardStepStamps'],res['iscores'],'o-',label=label)
        plt.ylabel('IS')
        plt.xlabel('num forward steps')

    return


def plot_a_scaling_result(fileStart,listOfFiles,label,pos):
    fsteps = []
    ws = []
    for file in listOfFiles:
        if type(file) is int:
            forwardStepsPerSecondPerGPU = 0
            worldsize = file
        else:

            with open(fileStart+file, 'rb') as handle:
                res = pickle.load(handle)

                total_forward_steps = res['forwardStepStamps'][-1]
                worldsize = res['world_size']
                totalActiveRuntime = res['timestamps'][-1]
                forwardStepsPerSecondPerGPU = total_forward_steps/totalActiveRuntime


        fsteps.append(forwardStepsPerSecondPerGPU)
        ws.append(str(worldsize))

    ind = np.arange(len(listOfFiles))
    width = 0.35


    plt.bar(ind+width*pos,fsteps,width,label=label)

    plt.xticks(ind+width/2,ws)




if __name__ == "__main__2":

    fileStart = "results/moreFilters/extragrad/"
    listOfFiles = ['eg8','eg16','eg32']
    label = "extragrad"
    plot_a_scaling_result(fileStart,listOfFiles,label,0)
    fileStart = "results/moreFilters/ps/"
    # list of files has to be in right order
    # if missing a result put the worldsize for that missing result
    listOfFiles = [8,'ps16','ps32']
    label = "ps"
    plot_a_scaling_result(fileStart,listOfFiles,label,1)
    plt.legend()
    plt.show()


if __name__ == "__main__1":
    calculate_scaling_result("psd/psd32_f")


    get_a_plot("psd/psd32_f","time","psd32",1)
    get_a_plot("psd/psd2","time","psd2",1)
    plt.grid()
    plt.legend()
    plt.show()




listOfAlgs = [
"extragrad/eg2GP",
"asyncEG/SPS2_GP",
"asyncEG/SPSp8_GP",
              ]

for alg in listOfAlgs:
    print_a_res(alg)
    print("\n\n\n")

for alg in listOfAlgs:
    get_a_plot(alg,"time",alg,2)


plt.grid()
plt.legend()
plt.show()

if __name__ == "__main__1":
    fileStart = "results/extragrad/"
    listOfFiles=['extragrad2','extragrad8','extragrad16','extragrad32']
    plot_a_scaling_result(fileStart,listOfFiles,'extragrad',0)
    plt.legend()
    plt.show()
