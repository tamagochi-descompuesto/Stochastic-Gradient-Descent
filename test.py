#Libraries and dependencies
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def main(WEIGHT, ALPHA, RATIO, iteration): 
    #Reading the data from csv
    print("----- READING DATA -----\n")
    sleep(2)
    data= pd.read_csv('data.csv', sep=',')
    
    X = data.iloc[:,:-1]
    print("-------- X VALUES -----")
    print(X, '\n')
    T = data.iloc[:,-1]
    print("----- TARGET COLUMN -----")
    print(T, '\n')
    N = len(X.index)
    
    train_rows = random.sample(range(N), int(RATIO*N//100))
    
    X_train = X.iloc[train_rows].to_numpy()
    X_test = X.drop(train_rows).to_numpy()
    print("----- TRAIN DATA -----")
    print(X.iloc[train_rows], '\n')

    T_train = T.iloc[train_rows].to_numpy()
    T_test = T.drop(train_rows).to_numpy()
    print("----- TRAIN TARGET COLUMN -----")
    print(T.iloc[train_rows], '\n')

    #Building the W array
    W = np.full(X_train.shape[1], WEIGHT)
    
    #Calculating the first o with the activation function that is sign(w0x0 + w1x1 + w2x2)
    o = np.sign(X_train @ W)

    #Initializing epoch number
    epoch = 0
    
    print('----- STARTING TRAINING------\n')
    sleep(2)
    #Entering the loop if condition doesn't applies
    while(not np.array_equal(o, T_train)):
        print("***** O *****")
        print(pd.DataFrame(o))
        print("***** T *****")
        print(pd.DataFrame(T_train))
        print("***** W *****")
        print(pd.DataFrame(W))

        error_vec = np.array(T_train - o)
        
        #Updating the weights
        for i in range(X_train.shape[0]):
            W += (X_train[i] * (ALPHA*error_vec[i]))

        print("\n***** W updated *****")
        print(pd.DataFrame(W))

        #Recalculating o and updating epoch
        o = np.sign(X_train @ W)
        epoch += 1
    
    #Printing final results
    print("\n----- TRAINING ENDED -----\n")
    print("***** O *****")
    print(pd.DataFrame(o))
    print("***** T *****")
    print(pd.DataFrame(T_train))
    print("***** X TRAINING *****")
    print(pd.DataFrame(X_train))
    print("***** W *****")
    print(pd.DataFrame(W))

    print('\n----- STARTING TEST -----\n')
    sleep(2)
    o_test = np.sign(X_test @ W)
    error = T_test - o_test
    acc = 100*np.count_nonzero(error == 0.0)/len(error)

    print('----- TESTING ENDED ------\n')
    print("***** X TEST *****")
    print(pd.DataFrame(X_test))
    print("***** O PREDICTED *****")
    print(pd.DataFrame(o_test))
    print("***** REAL T *****")
    print(pd.DataFrame(T_test))
    print("***** ERROR *****")
    print(error)
    print("***** MSE *****")
    mse = sum((error)**2)/len(T_test)
    print(mse)
    print("***** ACCURACY ******")
    print(acc)
    print("***** EPOCHS *****")
    print(epoch)

    plt.figure(figsize=(10, 10))
    plt.plot(o_test, label='Predictions')
    plt.plot(T_test, label='Real values')
    plt.title('Predictions vs. Real Values {iteration}'.format(iteration=iteration + 1))
    plt.legend()
    plt.show()

    return epoch, acc, mse


if __name__ == '__main__':
    weight_arr = [0.1, 0.1, 0.2, 0.3, 0.8, 0.9, 0.1, 0.1, 0.2, 0.1]
    alpha_arr = [0.5, 0.5, 0.1, 0.3, 0.9, 1, 0.1, 0.5, 0.5, 0.5]
    ratio_arr = [70, 70, 80, 90, 90, 90, 50, 40, 60, 80]

    epochs = []
    accs = []
    mse_arr = []

    for i in range(len(weight_arr)):
        epoch, acc, mse = main(weight_arr[i], alpha_arr[i], ratio_arr[i], i)
        epochs.append(epoch)
        accs.append(acc)
        mse_arr.append(mse)

    plt.scatter(ratio_arr, accs)
    plt.title('Train-test ratio vs accuracy')
    plt.xlabel('Ratio')
    plt.ylabel('Accuracy')
    plt.show()

    plt.scatter(weight_arr, accs)
    plt.title('Weights of the perceptron vs accuracy')
    plt.xlabel('Weight')
    plt.ylabel('Accuracy')
    plt.show()

    plt.scatter(alpha_arr, accs)
    plt.title('Alpha values vs accuracy')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.scatter(epochs, mse_arr)
    plt.title('Acumulated MSE throught the testings')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()

    plt.figure(figsize=(10,10))
    plt.scatter(epochs, accs)
    plt.title('Accuracy throught the testings')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()