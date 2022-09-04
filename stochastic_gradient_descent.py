#Libraries and dependencies
from time import sleep
import numpy as np
import pandas as pd
import random

def main(): 

    #Initializing weights with random values entered by the user
    WEIGHT = float(input('Enter the initial weights, please make it a number between 1 and 0: '))
    ALPHA = float(input('Enter the learning rate, a good value could be 0.5: ')) 
    RATIO = float(input('Enter the ratio of training and testing arrays, anumber between 10 and 100 in skips of 10 (10, 50, 70, etc.): '))

    #Reading the data from csv
    """
    - First column of the document represents X0 vector, second column X1 and so on
    - Last column represents the target vector t
    """
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
    print(sum((error)**2)/len(T_test))
    print("***** ACCURACY ******")
    print(acc)
    print("***** EPOCHS *****")
    print(epoch)


if __name__ == '__main__':
    main()  