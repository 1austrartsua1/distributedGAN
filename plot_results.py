

import pickle
import matplotlib.pyplot as plt

with open('results/fbf/results_12_04_2021::14:06:21', 'rb') as handle:
    res = pickle.load(handle)

print(f"total runtime = {res['total_running_time']/60:.4f} minutes")

print(res)

getScalingResults = False
if getScalingResults:
    # weak scaling results

    total_forward_steps = res['forwardStepStamps'][-1]
    worldsize = res['world_size']
    totalActiveRuntime = res['timestamps'][-1]
    forwardStepsPerSecondPerGPU = total_forward_steps/totalActiveRuntime


    print(f"algorithm={res['algorithm']}, worldsize={worldsize}")

    print(f"total_forward_steps = {total_forward_steps}")

    print(f"totalActiveRuntime={totalActiveRuntime}")
    print(f"forwardStepsPerSecondPerGPU={forwardStepsPerSecondPerGPU}")


plt.plot(res['timestamps'],res['iscores'],'o-')
plt.ylabel('IS')
plt.xlabel('time (s)')
plt.grid()



plt.show()

plt.plot(res['forwardStepStamps'],res['iscores'],'o-',label="batchsize=512")
plt.ylabel('IS')
plt.xlabel('num forward steps')

print(res['forwardStepStamps'][:10])

with open('results/fbf/fbf_replicate1', 'rb') as handle:
    res = pickle.load(handle)

plt.plot(res['genUpdateStamps'],res['iscores'],'o-',label="batchsize=64")
#plt.ylabel('IS')
#plt.xlabel('time (s)')
plt.grid()
plt.legend()
print(res['genUpdateStamps'][:10])

plt.show()
