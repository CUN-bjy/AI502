# 2. Find the best parameters with FCN model

# various num_epochs(sigmoid)
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-e100 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --num_epochs 100&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-e200 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --num_epochs 200&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-e500 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --num_epochs 500&

# various num_epochs(relu)
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-relu-e100 --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu --num_epochs 100&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-relu-e200 --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu --num_epochs 200&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-relu-e500 --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu --num_epochs 500&

# various learning_rate
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-lr5e4-e500 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --learning-rate 0.0005 --num_epochs 500&
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-lr1e3-e500 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --learning-rate 1e3 --num_epochs 500&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-lr5e3-e500 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --learning-rate 0.05 --num_epochs 500&

# various batch_size
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-b128 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --batch-size 128&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-b256 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --batch-size 256&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-b512 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --batch-size 512&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-b1024 --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --batch-size 1024&

# various optimizer
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-adam --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --optimizer adam&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-rms --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --optimizer rmsprop&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name fcn-L3-H512-sigmoid-sgd --mode FCN --num_layer 3 --hidden_size 512 --active_fn sigmoid --optimizer sgd&