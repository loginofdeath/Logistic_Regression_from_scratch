import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cross_validation import train_test_split
from preprocess import get_binary_data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#dataset = pd.read_csv('ecommerce_data.csv')

X,Y=get_binary_data()

# create train and test sets
Xtrain = X[:-100]
Ytrain = Y[:-100]
Xtest = X[-100:]
Ytest = Y[-100:]

# randomly initialize weights
D = X.shape[1]
W = np.random.randn(D)
b = 0 # bias term

# make predictions
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

# cross entropy
def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

learning_rate = 0.001

for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    # ctrain = cross_entropy(Ytrain, pYtrain)
    # ctest = cross_entropy(Ytest, pYtest)
    # train_costs.append(ctrain)
    # test_costs.append(ctest)

    # gradient descent
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    # if i % 1000 == 0:
    #     print i, ctrain, ctest	
    
#print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))

##################################

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Xtrain, Ytrain)

# Predicting the Test set results
y_pred_logistic = classifier.predict(Xtest)
from sklearn.metrics import accuracy_score

#accuracy_score(y_true, y_pred)
print("Final test classification_rate(using sklearn):", accuracy_score(Ytest, y_pred_logistic))


