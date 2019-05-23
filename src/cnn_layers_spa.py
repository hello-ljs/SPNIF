# -*- coding: utf-8 -*-
import numpy as np
from blob2matrix import *
from scipy import sparse as ss
from scipy.sparse import csr_matrix as cm

def convolution(input_feature,filters,stride_w,stride_h,pad_w,pad_h,*args):
    filters_matrix_spa = ss.load_npz(filters)
    filters_matrix = filters_matrix_spa.toarray()
    #filters_matrix = np.load(filters)
    N,C,H,W = input_feature.shape
    input_feature_matrix = data_2_matrix(input_feature,filters_matrix,stride_w,stride_h,pad_w,pad_h,*args)
    #filters_matrix = filter_to_2d(filters)
    #print (filters_matrix.shape)
    #np.save('conv1_params.npy',filters_matrix)
    #print (input_feature_matrix.shape)
    output_feature_matrix = np.dot(filters_matrix,input_feature_matrix)
    
    out_n = N
    out_c = args[0][0]
    out_h = (H + 2 * pad_w - args[0][3]) // stride_w + 1
    out_w = (W + 2 * pad_h - args[0][3]) // stride_h + 1

    output_feature = output_feature_matrix.reshape(out_n,out_c,out_h,out_w)
    return output_feature

def fc_layer(input_feature,filters):
    filters_matrix_spa = ss.load_npz(filters)
    filters_matrix = filters_matrix_spa.toarray()
    #filters_matrix = np.load(filters)
    if (input_feature.ndim == 2):
        fc_out = np.dot(input_feature,filters_matrix.T)
    else:
        N,C,H,W = input_feature.shape
        input_feature_matrix = input_feature.reshape(N,C*H*W)
        fc_out = np.dot(input_feature_matrix,filters_matrix.T)
    return fc_out

def relu(feature_map):
    #Preparing the output of the ReLU activation function.
    if (feature_map.ndim == 2):  # 全链接之后进行Relu
        relu_out = np.zeros(feature_map.shape)
        for h in range(feature_map.shape[0]):
            for w in range(0,feature_map.shape[1]):
                relu_out[h,w] = np.maximum(feature_map[h,w], 0)
        return relu_out
    else:                        #卷积之后进行Relu
        relu_out = np.zeros(feature_map.shape) 
        for n in range(feature_map.shape[0]):
            for c in range(0,feature_map.shape[1]):
                for h in range(0, feature_map.shape[2]):
                    for w in range(0, feature_map.shape[3]):
                        relu_out[n, c, h, w] = np.maximum(feature_map[n, c, h, w], 0)
        return relu_out

def max_pooling(input_feature, pooling_size, strides=(1, 1), padding=(0, 0)):
    """
    最大池化前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param pooling: 池化大小(k1,k2)
    :param strides: 步长
    :param padding: 0填充
    :return:
    """
    N, C, H, W = input_feature.shape

    padding_z = np.lib.pad(input_feature, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant', constant_values=0)


    out_h = (H + 2 * padding[0] - pooling_size[0]) // strides[0] + 1
    out_w = (W + 2 * padding[1] - pooling_size[1]) // strides[1] + 1

    pool_max = np.zeros((N, C, out_h, out_w))

    for n in np.arange(N):
        for c in np.arange(C):
            for i in np.arange(out_h):
                for j in np.arange(out_w):
                    pool_max[n, c, i, j] = np.max(padding_z[n, c,strides[0] * i:strides[0] * i + pooling_size[0],strides[1] * j:strides[1] * j + pooling_size[1]])
    return pool_max

def softmax(vector):
    return np.exp(vector)/np.sum(np.exp(vector))

'''def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum    
    return s'''
