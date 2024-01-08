import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

data1 = []
with open('./cifar_10_iid/random.txt', 'r') as file:
    for line in file:
        value = float(line.strip())
        data1.append(value)
m1 = make_interp_spline(range(50), data1)
xs1 = np.linspace(0, 49, 500)
y1 = m1(xs1)

data2 = []
with open('./cifar_10_iid/oort.txt', 'r') as file:
    for line in file:
        value = float(line.strip())
        data2.append(value)
m2 = make_interp_spline(range(50), data2)
xs2 = np.linspace(0, 49, 500)
y2 = m2(xs2)

data3 = []
with open('./cifar_10_iid/ours.txt', 'r') as file:
    for line in file:
        value = float(line.strip())
        data3.append(value)
m3 = make_interp_spline(range(50), data3)
xs3 = np.linspace(0, 49, 500)
y3 = m3(xs3)

plt.figure(figsize=(10, 6))
plt.rcParams.update({'font.size': 15})
plt.plot(xs1, y1, label='random', color='#3cb44b', linewidth=2.5)
# plt.plot(,data1, label='random')
plt.plot(xs2, y2, label='oort', color='#4363d8', linewidth=2.5)
# plt.plot(data2, label='oort')
plt.plot(xs3, y3, label='ours', color='#e6194B', linewidth=2.5)
# plt.plot(data3, label='ours')
plt.legend(prop={'size': 15}, loc='lower right')
plt.xlabel('epoch', fontsize=15)
plt.ylabel('ACC', fontsize=15)
# plt.title('Testing acc', fontsize=15)
plt.title('Testing acc iid MobileNet Cifar-10', fontsize=18)
# plt.savefig('non-iidα01.png')
plt.show()
