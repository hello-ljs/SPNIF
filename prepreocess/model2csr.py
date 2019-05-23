# -*- coding: utf-8 -*-
# writed by lollipoper1993 
# last modified in 2019 01 09

# Don't modify the code unless you know what you are doing!
import argparse
import numpy as np
from scipy import sparse as ss
from scipy.sparse import csr_matrix as cm
import caffe
from caffe import Net

import sys 
sys.path.append('../src/')
from blob2matrix import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--deploy_file",
    default='/home/sdu/SPNIF/models/caffemodels/MNIST/lenet.prototxt',
    help="-- deploy file path."
)
parser.add_argument(
    "--model_file",
    default='/home/sdu/SPNIF/models/caffemodels/MNIST/lenet.caffemodel',
    help="--caffemodel file path ."
)

parser.add_argument(
    "--output_dir",
    default='/home/sdu/SPNIF/models/convered_models/MNIST/',
    help="--output file path."
)
parser.add_argument(
    "--dtype",
    default=np.float32,
    help="--output file path."
)
args = parser.parse_args()



prototxt=args.deploy_file
caffe_model=args.model_file 
output_dir = args.output_dir
net = caffe.Net(prototxt,caffe_model, 0)
[(k,v[0].data.shape) for k,v in net.params.items()]  #查看各层参数规模





def get_threshold(layer_name):
    layer = net.params[layer_name][0].data
    threshold = np.median(abs(layer))
    return threshold

def blob2csr(layer_name,threshold):
    param = net.params[layer_name][0].data
    pm = filter_2_matrix(param)# param matirx
    counter = 0
    for i_n in range(pm.shape[0]):
        for i_c in range(pm.shape[1]):
            if abs(pm[i_n][i_c]) < 2*threshold:
                pm[i_n][i_c] = 0
                counter = counter +1
    spa_pm = cm(pm)
    ss.save_npz(output_dir + '{}'.format(layer_name),spa_pm,'a')
    zero_persent = (float(counter)/float(pm.size))*100
    print('zero weights percentage in layer {} is: '.format(layer_name) + str(zero_persent)+'%')
    return pm

def blob2matrix(layer_name):
    param = net.params[layer_name][0].data
    pm = filter_2_matrix(param)# param matirx
    
    np.save(output_dir + '{}'.format(layer_name),pm,'a')
    print ('done')

def main():
    #for k,v in net.params.items():
        #threshold = get_threshold(k)
    
        #process_layer(k,threshold)
    for k,v in net.params.items():
        blob2matrix(k)


if __name__ == '__main__':
   main()












