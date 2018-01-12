import numpy as np
from skimage import io
import os
import sys

mean = io.imread("average_Q1.jpeg")
mean = mean.flatten()
x_mean = []
for i in range(415):
    x_mean.append(mean)
x_mean = np.array(x_mean)
x_mean = np.transpose(x_mean)


data_file = sys.argv[1]
all_image = os.listdir(data_file)
data_set = []
for image_name in all_image:
    data_set.append(io.imread(data_file+"/"+image_name))

    
x = []
   
for image in data_set:
    image = image.flatten()
    x.append(image)

x = np.array(x)
x = np.transpose(x)


U, s, V = np.linalg.svd(x - x_mean, full_matrices=False)




image = io.imread(data_file+"/"+sys.argv[2]).flatten()
w0 = np.dot(image, U[: ,0])
w0 = np.dot(w0, U[: ,0])
w1 = np.dot(image, U[: ,1])
w1 = np.dot(w1, U[: ,1])
w2 = np.dot(image, U[: ,2])
w2 = np.dot(w2, U[: ,2])
w3 = np.dot(image, U[: ,3])
w3 = np.dot(w3, U[: ,3])
w = w0+w1+w2+w3+mean
w = w.astype(np.uint8)
w = np.reshape(w, (600,600,3))
io.imsave("reconstruction.png", w)


'''
# 3. 

Q3_data_set = []
Q3_data_set.append(data_set[26].flatten())
Q3_data_set.append(data_set[321].flatten())
Q3_data_set.append(data_set[324].flatten())
Q3_data_set.append(data_set[241].flatten())

i = 0
for image in Q3_data_set:
    w0 = np.dot(image, U[: ,0])
    w0 = np.dot(w0, U[: ,0])
    w1 = np.dot(image, U[: ,1])
    w1 = np.dot(w1, U[: ,1])
    w2 = np.dot(image, U[: ,2])
    w2 = np.dot(w2, U[: ,2])
    w3 = np.dot(image, U[: ,3])
    w3 = np.dot(w3, U[: ,3])
    w = w0+w1+w2+w3+mean
    w = w.astype(np.uint8)
    w = np.reshape(w, (600,600,3))
    io.imsave("reconstruction_"+str(i)+".png", w)
    i += 1
'''
'''
# 2. Draw eigenface
for i in range(4):
    eigenface = U[: ,i]
    eigenface -= np.min(eigenface)
    eigenface /= np.max(eigenface)
    eigenface = (eigenface*255).astype(np.uint8)
    eigenface = np.reshape(eigenface, (600,600,3))
    print(eigenface)
    io.imsave("eigenface_"+str(i)+".png", eigenface)
'''
    
    