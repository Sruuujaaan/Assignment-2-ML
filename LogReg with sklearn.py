import pandas as pd
import numpy as np
from sklearn import linear_model

# Random number seed for duplication purposes
np.random.seed(42)

# Inputs edited data file
# Removed titles, changed male to 0 and female to 1
dataFile = pd.ExcelFile('voiceDat.xlsx')

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
randDat_X = randDat[:, :-1]
randDat_Y = randDat[:, -1]

# Split the data into training/testing sets (divides the rows)
randDat_X_train = randDat_X[0:2534]
randDat_X_test = randDat_X[2534:]

# Split the targets into training/testing sets
randDat_Y_train = randDat_Y[0:2534]
randDat_Y_test = randDat_Y[2534:]

# Create linear regression object
# NOTE C is the Inverse of regularization strength
# smaller values specify stronger regularization
# IE we want it big so it doesn't regularize our weights
# Default iterations is 100
logreg = linear_model.LogisticRegression(C=1e5)

# Create linear regression object
logreg.fit(randDat_X_train, randDat_Y_train)

# Print the resulting weights
print "B0: ", logreg.intercept_ 
print "Coefficients: \n", logreg.coef_

predict_y = logreg.predict(randDat_X_test)
total = 0
for i in range(0, len(predict_y)):
    total += (predict_y[i] == randDat_Y_test[i])

print "total: ", total, "out of: ", len(predict_y), "correct"
print total/float(len(predict_y))*100, "% correct"

