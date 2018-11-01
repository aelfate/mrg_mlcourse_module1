# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import classification_report
from mnist import MNIST

x_train_dir = './data'
model_input_dir = 'data/'
mndata = MNIST(x_train_dir)
mndata.gz = True

X, y = mndata.load_training()

X = np.array(X)
y = np.array(y)

record_num, features_num = X.shape


def scale_data(X):
    X = X / 255
    
    return X

X = scale_data(X)
bias_f = np.ones((record_num, 1))
X = np.hstack((bias_f, X))
record_num, features_num = X.shape
y = np.array(y)

def last_predict(X, weights_vector):
    predict_y = []
    for elem in X:
        score = []
        for weights in weights_vector:
            score.append(np.sum(elem*weights))
        score = np.array(score)
        i, = np.where(score == np.max(score))
        predict_y.append(int(i))
    return predict_y

weights = np.load(model_input_dir + 'model_weights.npy')

predicted_value = last_predict(X, weights)

print(classification_report(np.array(y), predicted_value))