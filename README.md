# DeepID2_based_Inception2
this network is built based on Inception. and optimize it by using identification loss and verification loss which derive from DeepID2+.
the project files are place in RootCaffe/model/DeepID2_based_Inception/ and RootCaffe/python/pydata.py

# config
1. you should compile caffe firstly

# recommendation
It is difficult to optimize the verification loss at the beginning stages. So I only optimize the identification loss at the beginning stages. And the verification loss are joined after the network can work for identification task.
