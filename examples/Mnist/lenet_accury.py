# -*- coding: utf-8 -*-
import numpy as np
import sys 
sys.path.append('../../src/')
import time
from blob2matrix import *
from cnn_layers import *
import cv2
import os

import json

def load_json(dic_dir):
    with open (dic_dir,'r') as r:
        shape_dict = json.load(r)
    return (shape_dict)

images_dir = '/home/sdu/SPNIF/images/numbers/'
json_dir = '/home/sdu/SPNIF/models/convered_models/MNIST/shape_dic.json'
shape_dict = load_json(json_dir)

filter_conv1='/home/sdu/SPNIF/models/convered_models/MNIST/fp32/conv1.npy'
filter_conv2='/home/sdu/SPNIF/models/convered_models/MNIST/fp32/conv2.npy'
filter_ip1='/home/sdu/SPNIF/models/convered_models/MNIST/fp32/ip1.npy'
filter_ip2='/home/sdu/SPNIF/models/convered_models/MNIST/fp32/ip2.npy'

def main():
    counter = 0
    for label in os.listdir(images_dir):# label is number such as 0,1,2...
        img_path = os.path.join(images_dir, label)
        #print img_path
        for path,dirnames,filenames in os.walk(img_path):
            for filename in filenames:
                img_path = path +'/'+filename
                test_img = cv2.imread(img_path)
                test_img = cv2.cvtColor(test_img,cv2.COLOR_RGB2GRAY)
                test_img = test_img.astype(np.float32)
                test_img = test_img.reshape(1,1,28,28)
                test_img = test_img /255
            
                conv1_out = convolution(test_img,filter_conv1,1,1,0,0,shape_dict['conv1'])
                pool1_out = max_pooling(conv1_out,(2,2),strides=(2,2))
            	conv2_out = convolution(pool1_out,filter_conv2,1,1,0,0,shape_dict['conv2'])
            	pool2_out = max_pooling(conv2_out,(2,2),strides=(2,2))
            	ip1_out = fc_layer(pool2_out,filter_ip1)
            	relu1_out = relu(ip1_out)
            	ip2_out = fc_layer(relu1_out,filter_ip2)
            	result = softmax(ip2_out)
            	result = result.argmax()
           	if(int(label) == result):
            	    counter = counter + 1 
    print ('Accuracy is' +  str(float(counter/100.0))+'%')

if __name__ == '__main__':
    sys.exit(main())
