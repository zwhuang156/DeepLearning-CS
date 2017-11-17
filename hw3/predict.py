import os
import numpy as np
import pandas as pd
import csv
import sys
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.models import load_model


def Load_testing_data(test_data_path):
    test_data = pd.read_csv(test_data_path, sep=',', header=0)
    X_test = test_data['feature']
    X_test = X_test.str.split(' ', expand=True)
    X_test = np.array(X_test.values)
    X_test = np.reshape(X_test, (X_test.shape[0], 48, 48, 1))
    
    return X_test
    

if __name__ == '__main__':

    model_path = "model/model.h5"
    test_data_path = sys.argv[1]
    ans_path = sys.argv[2]

    model = load_model(model_path)
    X_test = Load_testing_data(test_data_path)
    
    result = model.predict(X_test)
    
    ans = []
    for i in range(result.shape[0]):
        ans.append(np.argmax(result[i]))
    
    
    # Output answer
    output = open(ans_path, "w+")
    s = csv.writer(output,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(ans)):
        s.writerow([str(i), str(ans[i])]) 
    output.close()
    
    print("Answer was written !!")