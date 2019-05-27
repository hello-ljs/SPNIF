# Sample Network Inference Framework
## What Is SPNIF 
   SPNIF is a python framework that convert the Neural Network models into embeded-friendly format.  
   It is writen in python and use numpy to do the matrix computing.  
   The C++ version SPNIF is undering development.  
## Advantages
   Cuting out 60% of the parameters and computational complexity  by pruning the network without lossing accuracy.  
   Using sparse matrix to store parameter-matrix and saving 2/3 Storage consumption.  
   Converting FP32 models into int8 models and that further saves three quarters of the Storage consumption.  
## Supported Framworks and layers
   Now SPNIF only supports caffemodels. Pytorch and TF models are unding development.
   SPNIF supports  
   2D convlutions 
   Relu 
   softmax 
   fc/ip 
## Dependence environment 
   Anaconda2  
   numpy  
   scipy 
## How to use 
   step 1:use preprocess/prune_weights.ipynb to prune the networks  
   step 2:use preprocess/model2csr.py to transform models   
   step 3:use exmaples/Mnist/lenet.py to do the inference. 
   
## Performance
   SPNIF takes 138 seconds to infer the lenet5 10,000 times.  
   The average inferencing time is 13.8 ms per image.
