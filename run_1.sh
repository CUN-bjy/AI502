# 1. Find the best architecture of FCN model

# various num_layer(sigmoid)
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L5-H512-sigmoid --mode FCN --num_layer 5 --hidden_size 512 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L7-H512-sigmoid --mode FCN --num_layer 7 --hidden_size 512 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L9-H512-sigmoid --mode FCN --num_layer 9 --hidden_size 512 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L11-H512-sigmoid --mode FCN --num_layer 11 --hidden_size 512 --active_fn sigmoid&

# various num_layer(relu)
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-relu --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L5-H512-relu --mode FCN --num_layer 5 --hidden_size 512 --active_fn relu&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L7-H512-relu --mode FCN --num_layer 7 --hidden_size 512 --active_fn relu&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L9-H512-relu --mode FCN --num_layer 9 --hidden_size 512 --active_fn relu&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L11-H512-relu --mode FCN --num_layer 11 --hidden_size 512 --active_fn relu&

# various num_layer(tanh)
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-tanh --mode FCN --num_layer 3 --hidden_size 512 --active_fn tanh&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L5-H512-tanh --mode FCN --num_layer 5 --hidden_size 512 --active_fn tanh&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L7-H512-tanh --mode FCN --num_layer 7 --hidden_size 512 --active_fn tanh&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L9-H512-tanh --mode FCN --num_layer 9 --hidden_size 512 --active_fn tanh&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L11-H512-tanh --mode FCN --num_layer 11 --hidden_size 512 --active_fn tanh&

# various hidden_size
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H64-sigmoid --mode FCN --num_layer 3 --hidden_size 64 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H128-sigmoid --mode FCN --num_layer 3 --hidden_size 128 --active_fn sigmoid&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H256-sigmoid --mode FCN --num_layer 3 --hidden_size 256 --active_fn sigmoid&
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H1024-sigmoid --mode FCN --num_layer 3 --hidden_size 1024 --active_fn sigmoid
