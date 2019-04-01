import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0.5, 1, 50)
y1 = x**2 - 2*x + 1
y2 = - np.log(x)

plt.figure(1,facecolor='white')
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--', label='re')
plt.plot(x, y2, color='blue', linewidth=1.0, label='hhh')
plt.xlim([0,1])
plt.show()
