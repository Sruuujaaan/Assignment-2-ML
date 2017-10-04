import pandas as pd
import numpy as np
from sklearn import linear_model

# Number of iterations and learning rate constants
iterate = 20000
alpha = .0000004

# Define sigmoid function
def sigmoid(x):
    x = np.clip( x, -500, 500 )
    return 1 / (1 + np.exp(-x))

def predict(X, coef):
    temp = sigmoid(np.dot(X,coef))
    for i in range(0,len(temp)):
        if(temp[i] < .5):
            temp[i] = 0
        else:
            temp[i] = 1
    return temp

# Random number seed for duplication purposes
np.random.seed(42)

# Inputs edited data file
# Removed titles, changed male to 0 and female to 1
dataFile = pd.ExcelFile('voiceDat2.xlsx')

#Converts the entire file to a numpy array
df = dataFile.parse('voice').iloc[:,:].values

# Data is organized, we need to randomize it to generate a valid training/test sets
# Generates random indices, no duplicate indices
indices = np.random.choice(df.shape[0], df.shape[0], replace=False)

# Uses the indices to create a randomized version of the input array
# Indices are used to reorder the rows
randDat = np.empty((df.shape[0], df.shape[1]))
for index in indices:
    randDat[index] = df[indices[index]]

# Change the values here to change which columns are x's/y's
randX = randDat[:, :-1]
randY = randDat[:, -1]

# Split the data into training/testing sets (divides the rows)
trainX = randX[0:2534]
testX = randX[2534:]

# Split the targets into training/testing sets
trainY = randY[0:2534]
testY = randY[2534:]

# Initialize weights
weight = np.zeros(randX.shape[1])
new_Weight = weight

for n in range(0, iterate):
    gradient = (sigmoid(trainX.dot(weight)) - trainY).dot(trainX)
    weight = weight - alpha*gradient
##    print "type(trainX): ", type(trainX)
##    print "type(sigmoid(trainX.dot(weight))): ", type(sigmoid(trainX.dot(weight)))
##    print "type(gradient): ", type(gradient)
##    print "trainX.shape: ", trainX.shape
##    print "sigmoid(trainX.dot(weight)).shape: ", sigmoid(trainX.dot(weight)).shape
##    print "trainY.shape: ", trainY.shape
##    print "gradient.shape: ", gradient.shape
##    print "gradient: ", gradient
##    print "weight: ", weight
##    print "absolute error: ", sum(abs(sigmoid(trainX.dot(weight)) - trainY))

print weight

predict_y = predict(testX, weight)
total = 0
for i in range(0, len(predict_y)):
    total += (predict_y[i] == testY[i])

print "total: ", total, "out of: ", len(predict_y), "correct"
print total/float(len(predict_y))*100, "% correct"

# np.set_printoptions(threshold=np.inf)
# print (sigmoid(trainX.dot(weight)) - trainY)

##[ 11.99667199]
##Coefficients: 
##[[ -2.42558373e+00  -3.83087121e+00   6.00738108e+00   1.85847718e+01
##   -2.05176382e+01  -3.91024100e+01  -1.74466953e-01   9.08136397e-03
##   -3.67890114e+01   1.01786466e+01  -1.40768443e+00  -2.42558373e+00
##    1.61214563e+02  -2.96819789e+01   3.27896368e+00  -6.97667975e-02
##   -9.81657391e-01  -4.86419455e-01   4.95237936e-01   4.29585676e+00]]
    
