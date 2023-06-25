import matplotlib.pyplot as plt
import numpy as np

train = np.loadtxt("./data_liner.csv" , delimiter="," , skiprows= 1)
train_x_y = train[:,0:2]
train_0_1 = train[:,2]

plt.plot(train_x_y[train_0_1 == 1 , 0] , train_x_y[train_0_1 == 1 , 1] , "o")
plt.plot(train_x_y[train_0_1 == 0 , 0] , train_x_y[train_0_1 == 0 , 1] , "x")
plt.show()
