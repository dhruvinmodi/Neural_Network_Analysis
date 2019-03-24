"""
Name: Dhruvin Modi
E-Mail: dhruvinmodi2015@gmail.com
"""
import matplotlib.pyplot as plt
import numpy as np
y = []
x = [p for p in np.arange(-10.0,10.0,0.1)]
x = np.asarray(x)
for i in x:
	y.append(1.0 / (1.0 + np.exp(-i)))
plt.scatter(x,y,s = 5)
plt.xlabel('Values in range of -10 to 10')
plt.ylabel('Sigmoid value')
plt.title('Sigmoid function')
plt.savefig('./Sigmoid.png')
plt.close()
