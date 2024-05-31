from nn import NN
import os
import numpy as np
import random
import cv2 

PATH = "trainingSet"
EPOCHS = 10
labels = os.listdir(PATH) 
w, h = 28,28

nn = NN( w*h, 10, 10, .1 )
data = []

def getY( x ):
    return np.array([ int(x==i) for i in range(10) ]).reshape(-1,1)

for label in labels:
    path = os.path.join( PATH, label )
    img_files = os.listdir( path )
    for img_name in img_files:
        file_path = os.path.join( path, img_name )
        data.append({
            "file" : file_path,
            "label" : label,
            "Y" : getY(int(label))
        })

random.shuffle( data )

for e in range(EPOCHS):
    for i, img_data in enumerate(data) :
        ip = cv2.imread( img_data["file"] )
        ip = cv2.cvtColor( ip, cv2.COLOR_BGR2GRAY )
        ip = ip.reshape( w*h, 1 ) / 255.0

        nn.train( ip, img_data["Y"], m=42000 )
        # if i%1000 == 0 :
            # print( nn.get_accuracy( ip, img_data["Y"]) )
            # nn.save("basic.nn")



nn.save("basic.nn")