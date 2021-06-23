# show the results of all methods for CIFAR-100

export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64

python utils/vis.py --dataset imagenet --resolution 64x64 --order_subset 10x10 --network resnet18 --num_orders 3
