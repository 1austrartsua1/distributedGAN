

import pickle
import matplotlib.pyplot as plt

with open('results/fbf_replicate1', 'rb') as handle:
    res = pickle.load(handle)

print(res)


plt.plot(res['timestamps'],res['iscores'],'o-')
plt.ylabel('IS')
plt.xlabel('time (s)')
plt.show()

plt.plot(res['genUpdateStamps'],res['iscores'],'o-')
plt.ylabel('IS')
plt.xlabel('num generator updates')
plt.show()
