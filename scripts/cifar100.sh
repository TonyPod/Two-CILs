# run all methods for CIFAR-100

export CUDA_VISIBLE_DEVICES=2
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cudnn-7.6/lib64
NUM_EXEMPLARS=20
COMMON_FLAGS="--network resnet"

for i in {1..5}; do

  # base model (1st class increment)
  python main.py --order_idx $i --dataset cifar100 --memory_type none --bias_rect none --reg_type none --base_model $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type none --bias_rect none --reg_type none --base_model --no_final_relu $COMMON_FLAGS

  # lowerbound
  python main.py --order_idx $i --dataset cifar100 --memory_type none --bias_rect none --reg_type none $COMMON_FLAGS

  # hybrid1
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect none --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # Class Imbalanced Learning techniques
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect post_scaling --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS --stat_boundary --geometric_view
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect random_undersampling --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect random_oversampling --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect reweighting --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect kmeans --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect kmedoids --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect adasyn --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect smote --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # iCaRL (CVPR'17)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect none --embedding --test_cosine --no_final_relu --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # EEIL (ECCV'18)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect eeil --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # IL2M (ICCV'19)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type none --bias_rect il2m --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # LSIL (CVPR'19)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect bic --bias_aug --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # MDFCIL (CVPR'20)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect weight_aligning_no_bias --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS --stat_boundary --geometric_view
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type lwf --bias_rect weight_aligning --adjust_lwf_w --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # GDumb (ECCV'20)
  python main.py --order_idx $i --dataset cifar100 --memory_type episodic --reg_type none --bias_rect random_undersampling --init_type all --num_exemplars $NUM_EXEMPLARS $COMMON_FLAGS

  # joint training
  python main.py --order_idx $i --dataset cifar100 --memory_type none --reg_type none --bias_rect none --joint_training $COMMON_FLAGS
done
