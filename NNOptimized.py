# This script loads the dataset and implements the Logistic Regression Classifier
# Load the important stuff
import numpy as np
import pickle
import pandas as pd

# Read Training, validation and test sets from the dump in DataGen.py in assignment 1
Xtrain, ytrain, Xval, yval, Xtest, ytest = pickle.load(open("Important2.pk", "rb"))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.array(x, dtype=np.float32)))

# Reshape all the training sets as matrix or vectors of data type float32

Xtrain = Xtrain.values.astype(np.float32)    # size 20000X2000
Xval = Xval.values.astype(np.float32)  # size 5000X2000
Xtest = Xtest.values.astype(np.float32)  # size 25000X2000
ytrain = np.asarray(ytrain,dtype=np.float32)  # turn ytrain into a 1D matrix
ytrain = np.reshape(ytrain, (20000, 1))  # turn ytrain into a 20000X1 matrix instead of the default 20000X0

yval = np.asarray(yval,dtype=np.float32)
yval = np.reshape(yval, (5000, 1))  # turn into a 5000X1 matrix

ytest = np.asarray(ytest,dtype=np.float32)
ytest = np.reshape(ytest, (25000, 1))  # turn into a 25000X1 matrix/vector

# initialize model parameters
W1 = np.random.uniform(-0.5, 0.5, (2000, 200)).astype(np.float32)  # Initialize W1: 2000X200
W2 = np.random.uniform(-0.5, 0.5, (200, 1)).astype(np.float32) # Initialize W2: 200X1

b1 = np.random.uniform(-0.5, 0.5, 200).astype(np.float32)  # Initialize b1: 200X1
b1 = np.reshape(b1, (200, 1))
b2 = np.random.uniform(-0.5, 0.5, 1).astype(np.float32) # scalar
b2 = np.reshape(b2, (1, 1))  # 1X1
ones=np.reshape(np.ones((20, 1)), (20, 1)) # For derivative of 1s

#learning rate initialization
alpha = np.float32(0.1)

# initialize variables to hold best model parameters and validation accuracy
best_accuracy = 0
best_W2 = W2
best_b2 = b2
best_W1 = W1
best_b1 = b1

#Initialize dataframe for learning curve
names = ["epoch", "Validation Accuracy", "Training Accuracy"]
df = pd.DataFrame(columns=names)


for epoch in range(0, 300):  # For each epoc
    #print(epoch)

    for j in range(0, 20000, 20):  # Batch forward and back propagation
        #Create batch
        Xbatch = Xtrain[j:j + 20]  # 20  X 2000
        ybatch = ytrain[j:j + 20]  # 20  X 1

        #Forward Propagation
        y1 = sigmoid(Xbatch.dot(W1) + b1.T)  # 20  X 200  Step 1 of FP
        y2 = sigmoid(y1.dot(W2) + b2)  # 20  X 1          Step 2 of FP

        #BackPropagation
        #Step 1
        dJ_dz2 = y2 - ybatch #20X1
        dJ_dW2 = y1.T.dot(dJ_dz2) / 20  # 200  X 1
        dJ_db2 = dJ_dz2.T.dot(ones) / 20  #1X1
        #Step 2
        dJ_dz1 = dJ_dz2.dot(W2.T) * y1 * (1 - y1)  # 20LX200
        dJ_dW1 = Xbatch.T.dot(dJ_dz1) / 20  # 2000X200
        dJ_db1 = dJ_dz1.T.dot(ones) / 20         # 200X1

        #Gradient Descent
        W2 = W2 - alpha * dJ_dW2
        b2 = b2 - alpha * dJ_db2
        W1 = W1 - alpha * dJ_dW1
        b1 = b1 - alpha * dJ_db1


    #Calculate validation accuracy
    yhat = np.where(sigmoid(sigmoid(Xval.dot(W1) + b1.T).dot(W2) + b2) >= 0.5, 1, 0)
    vaccuracy = np.sum(np.where(yhat == yval, 1, 0)) / float(len(yval)) * 100
    print("Validation Accuracy is:")
    print(vaccuracy)

    # Calculate Training accuracy
    yhat = np.where(sigmoid(sigmoid(Xtrain.dot(W1) + b1.T).dot(W2) + b2) >= 0.5, 1, 0)
    accuracy = np.sum(np.where(yhat == ytrain, 1, 0)) / float(len(ytrain)) * 100
    print("Training Accuracy is:")
    print(accuracy)

    #Add training and validation accuracy for epoch in dataframe
    ajaira = [epoch, vaccuracy, accuracy]
    df = df.append(pd.DataFrame([ajaira], columns=names))

    #Save best model parameters so far
    if (vaccuracy > best_accuracy):
        best_accuracy = vaccuracy
        best_W2 = W2
        best_b2 = b2
        best_W1 = W1
        best_b1 = b1

#Calculate Test Accuracy
yhat = np.where(sigmoid(sigmoid(Xtest.dot(best_W1) + best_b1.T).dot(best_W2) + best_b2) >= 0.5, 1, 0)
test_accuracy = np.sum(np.where(yhat == ytrain, 1, 0)) / float(len(ytest)) * 100
print('Test accuracy is:')
print(test_accuracy)

#Export to csv for learning curve
export_csv = df.to_csv('LearningCurveNN.csv', index=None, header=True)
