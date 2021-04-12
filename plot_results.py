

import pickle
import matplotlib.pyplot as plt

with open('results/results_12_04_2021::08:33:34', 'rb') as handle:
    res = pickle.load(handle)

print(res)

# strong scaling results
#samples processed per second
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
plt.show()

plt.plot(res['forwardStepStamps'],res['iscores'],'o-')
plt.ylabel('IS')
plt.xlabel('num forward steps')
plt.show()
