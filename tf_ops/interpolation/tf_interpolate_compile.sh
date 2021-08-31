#/bin/bash
#
# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-10.0/include -lcudart -L /usr/local/cuda-10.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# TF1.4
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-10.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# Obtain cuda version (assume that nvcc has been added to path)
set -e
NVCC_VER=`nvcc --version`
NVCC_VER=`echo ${NVCC_VER} | cut -d "," -f 2`
NVCC_VER=`echo ${NVCC_VER} | cut -d " " -f 2`

# set tensorflow version
TF_VER=`python -c "import tensorflow as tf; print(tf.__version__)"`
TF_VER=${TF_VER:0:1}

TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# TF1.2
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC \
-I ${TF_INC} \
-I ${TF_INC}/external/nsync/public \
-I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ \
-L ${TF_LIB} -l:libtensorflow_framework.so -O2 -D_GLIBCXX_USE_CXX11_ABI=1

# Toggle USE_CXX11_ABI to 0 if there are include errors.
