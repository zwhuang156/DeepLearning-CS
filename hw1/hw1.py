import csv 
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import time

useful_data = [4,5,6,9,12,14,17]

w = np.load('68_model.npy')


#read testing data (投影片p11)
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

    
for r in row:
    if n_row%18==0:
        test_x.append([])
    if (n_row%18) in useful_data:
        for i in range(2,11):
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
        # PM2.5 **2
        """if n_row%18==9:
            for i in range(2,11):
                test_x[n_row//18].append((float(r[i]))**2)"""
    n_row = n_row+1
text.close()
test_x = np.array(test_x)

# add square term
test_x2 = test_x**2
test_x3 = test_x**3
test_x = np.concatenate((test_x,test_x2), axis=1)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)




#get ans.csv with your model (投影片p11續)
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()