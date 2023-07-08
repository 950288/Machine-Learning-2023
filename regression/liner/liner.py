import numpy as np
import matplotlib.pyplot as plt

# Read the train data from the file
train = np.loadtxt('./regression/liner/liner.csv' , delimiter="," , skiprows=1)
train_x = train[:,0] # [:,0] means all rows and column 0
train_y = train[:,1]

# Define a function
# Fn(x) = theta0 + theta1 * x
theta0 = np.random.rand() # random number between 0 and 1
theta1 = np.random.rand()
# theta0 = 231.50
# theta1 = 1.395
def Fn(x):
    return theta0 + theta1 * x

# Define a function to calculate the error
def Error(x , y):
    return 0.5 * np.sum((y - Fn(x)) ** 2)

# Update expression
# theta0 = theta0 - eta * sum(Fn(x) - y)
# theta1 = theta1 - eta * sum((Fn(x) - y) * x)

# Define a function to calculate the gradient
def Gradient(x , y):
    # theta0
    theta0_g = np.sum(Fn(x) - y)
    # theta1
    theta1_g = np.sum((Fn(x) - y) * x)
    return theta0_g , theta1_g

# Define a function to update the parameters
def Updata_Parameters(x , y):
    global theta0 , theta1
    eta = 0.00005/train.shape[0] # train.shape[0] means the number of rows
    theta0_g , theta1_g = Gradient(x , y)
    theta0 = theta0 - eta * theta0_g
    theta1 = theta1 - eta * theta1_g

# Loop to update the parameters
for i in range(10000):
    Updata_Parameters(train_x , train_y)
    if i % 10 == 0:
        print("Epoch:" , i , "theta0:" , theta0 , "theta1:" , theta1 ,"Error:" , Error(train_x , train_y))

print("theta0:" , theta0 , "theta1:" , theta1)
# Show the data and the function
plt.plot(train_x , train_y , 'o')
x = np.linspace(0 , 300 , 100)
plt.plot(train_x , Fn(train_x))
plt.show()



