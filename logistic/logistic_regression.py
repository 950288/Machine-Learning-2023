import matplotlib.pyplot as plt
import numpy as np

train = np.loadtxt("./data_quadratic.csv" , delimiter="," , skiprows= 1)
train_x_y = train[:,0:2]
train_0_1 = train[:,2]

# generate random theta
theta = np.random.rand(3)

def standardize(x):
    return (x - x.mean()) / x.std()

train_x_y = standardize(train_x_y)

plt.plot(train_x_y[train_0_1 == 1 , 0] , train_x_y[train_0_1 == 1 , 1] , "o")
plt.plot(train_x_y[train_0_1 == 0 , 0] , train_x_y[train_0_1 == 0 , 1] , "x")
# plt.show()
