# show the results of all methods for CIFAR-100

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64

python utils/vis.py --dataset cifar100 --network resnet --num_orders 5
