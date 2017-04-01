# DeepID2_based_Inception
this network is built based on Inception. and optimize it by using identification loss and verification loss which derive from DeepID2+.
the project files are place in RootCaffe/model/DeepID2_based_Inception/ and RootCaffe/python/pydata.py

# config
1. you should compile caffe with WITH_PYTHON_LAYER := 1 in makefile.config firstly 

# recommendation
It is difficult to optimize the verification loss at the beginning stages. So I only optimize the identification loss at the beginning stages. And the verification loss are joined after the network can work for identification task.


# Reference
[1] Y. Sun, X. Wang, X. Tang, Deeply learned face representations are sparse, selective, and robust

[2] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, A. Rabinovich, Going deeper with convolutions

[3] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, T. Darrell, Caffe: Convolutional architecture for fast feature embedding
