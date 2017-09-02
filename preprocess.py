import pandas as pd
import numpy as np

def get_data():
    df = pd.read_csv('ecommerce_data.csv')

    data = df.as_matrix()

    X = data[:,:-1]
    Y = data[:,-1]

    # normalize columns 1 and 2
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
     # create a new matrix X2 with the correct number of columns
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:,0:(D-1)] = X[:,0:(D-1)] 

    # one-hot
    for n in range(N):
        t = int(X[n,D-1])
        X2[n,t+D-1] = 1

    return X2, Y


def get_binary_data():
    # return only the data from the first 2 classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2