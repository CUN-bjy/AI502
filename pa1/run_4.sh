# 4. Find the best parameters with CNN model

# # various num_epochs(relu)
# # CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-e100 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --num_epochs 100&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-e200 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --num_epochs 200&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-e500 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --num_epochs 500&

# various learning_rate
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-lr5e4-e500 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --learning-rate 0.0005 --num_epochs 500&
# CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-lr1e3-e500 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --learning-rate 1e3 --num_epochs 500&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-lr5e3-e500 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --learning-rate 0.005 --num_epochs 500&

# various batch_size
# # CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-b128 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --batch-size 128&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-b256 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --batch-size 256&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-b512 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --batch-size 512&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-b1024 --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --batch-size 1024&

# # various optimizer
# # CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-adam --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --optimizer adam&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-rms --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --optimizer rmsprop&
CUDA_VISIBLE_DEVICES=0 python pa_1.py --exp-name CNN-L3-H5-relu-sgd --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu --optimizer sgd&