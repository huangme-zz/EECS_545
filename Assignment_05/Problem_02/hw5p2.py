from matplotlib import pyplot as plt
import numpy as np

x_1_list = []
x_2_list = []

x_1 = 0
x_1_list.append(x_1)
x_2 = np.random.normal(0.5, np.sqrt(0.75))
x_2_list.append(x_2)

while (len(x_2_list) != 5000):
  x_1 = np.random.normal((x_2 + 1) / 2.0, np.sqrt(0.75))
  x_1_list.append(x_1)
  x_2 = np.random.normal((x_1 + 1) / 2.0, np.sqrt(0.75))
  x_2_list.append(x_2)

# Plot
plt.hist(x_1_list, normed=True)
x = np.array(range(-2500,5000))/1000.0
y = np.exp(-(x-1)**2/2.0)/np.sqrt(2*np.pi)
plt.plot(x, y)
plt.savefig('hw5p2_1.png')
plt.close()

plt.hist(x_2_list, normed=True)
x = np.array(range(-2500,5000))/1000.0
y = np.exp(-(x-1)**2/2.0)/np.sqrt(2*np.pi)
plt.plot(x, y)
plt.savefig('hw5p2_2.png')