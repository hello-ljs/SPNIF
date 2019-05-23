# -*- coding: utf-8 -*-
import numpy as np
import sys 
sys.path.append('../../src/')
import time
from blob2matrix import *
from cnn_layers_spa import *
import cv2

import json
from multiprocessing import Process, Queue
import multiprocessing
import threading



def load_json(dic_dir):
    with open (dic_dir,'r') as r:
        shape_dict = json.load(r)
    return (shape_dict)
json_dir = '/home/sdu/SPNIF/models/convered_models/MNIST/shape_dic.json'
shape_dict = load_json(json_dir)




def image_process_worker(image_queue, n, c, h, w):
    for i in range(1, 1 + 10000):
        image = cv2.imread('/home/sdu/SPNIF/images/numbers/8/00061.png')
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) #https://blog.csdn.net/mrr1ght/article/details/80081657 caffe's channel problem 
        image = image.astype(np.float32)
        image = image.reshape(n,c,h,w)
        image_queue.put(image)

        #image = cv2.resize(image, (w, h))
        #image = image.transpose((2, 0, 1))
        #image = image.reshape((n, c, h, w))
        #image_queue.put(image)
    image_queue.put(None)


def async_infer_worker(image_queue,filter_conv1,filter_conv2,filter_ip1,filter_ip2):
    while (True):
        img = image_queue.get()
        if type(img) != np.ndarray:
                break
        conv1_out = convolution(img,filter_conv1,1,1,0,0,shape_dict['conv1'])
        pool1_out = max_pooling(conv1_out,(2,2),strides=(2,2))
        conv2_out = convolution(pool1_out,filter_conv2,1,1,0,0,shape_dict['conv2'])
        pool2_out = max_pooling(conv2_out,(2,2),strides=(2,2))
        ip1_out = fc_layer(pool2_out,filter_ip1)
        relu1_out = relu(ip1_out)
        ip2_out = fc_layer(relu1_out,filter_ip2)
        #print (ip2_out)
    duration = time.time() - start_time
    print ('inference 10000 images done' + 'total time constum is ' + str(duration) + 's ')

def main():
    global start_time
    start_time = time.time()

    filter_conv1='/home/sdu/SPNIF/models/convered_models/MNIST/conv1.npz'
    filter_conv2='/home/sdu/SPNIF/models/convered_models/MNIST/conv2.npz'
    filter_ip1='/home/sdu/SPNIF/models/convered_models/MNIST/ip1.npz'
    filter_ip2='/home/sdu/SPNIF/models/convered_models/MNIST/ip2.npz'
    
    image_queue = multiprocessing.Queue(maxsize= 4)
  


    preprocess_process = None
    preprocess_process = multiprocessing.Process(target=image_process_worker, args=(image_queue, 1, 1, 28, 28))
    preprocess_process.start()


    async_infer_worker(image_queue,filter_conv1,filter_conv2,filter_ip1,filter_ip2)
    preprocess_process.join()


if __name__ == '__main__':
    sys.exit(main())
