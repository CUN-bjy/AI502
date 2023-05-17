# 3. Find the best architecture of CNN model

# various num_layer_conv(sigmoid)
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H5-sigmoid --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv sigmoid&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L5-H5-sigmoid --mode CNN --num_layer_conv 5 --hidden_size_conv 5 --active_fn_conv sigmoid&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L7-H5-sigmoid --mode CNN --num_layer_conv 7 --hidden_size_conv 5 --active_fn_conv sigmoid&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L9-H5-sigmoid --mode CNN --num_layer_conv 9 --hidden_size_conv 5 --active_fn_conv sigmoid&

# various num_layer_conv(relu)
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H5-relu --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L5-H5-relu --mode CNN --num_layer_conv 5 --hidden_size_conv 5 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L7-H5-relu --mode CNN --num_layer_conv 7 --hidden_size_conv 5 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L9-H5-relu --mode CNN --num_layer_conv 9 --hidden_size_conv 5 --active_fn_conv relu&

# various num_layer_conv(tanh)
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H5-tanh --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L5-H5-tanh --mode CNN --num_layer_conv 5 --hidden_size_conv 5 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L7-H5-tanh --mode CNN --num_layer_conv 7 --hidden_size_conv 5 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L9-H5-tanh --mode CNN --num_layer_conv 9 --hidden_size_conv 5 --active_fn_conv tanh&

# various hidden_size_conv(relu)
# CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H5-relu --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H10-relu --mode CNN --num_layer_conv 3 --hidden_size_conv 10 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H20-relu --mode CNN --num_layer_conv 3 --hidden_size_conv 20 --active_fn_conv relu&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H30-relu --mode CNN --num_layer_conv 3 --hidden_size_conv 30 --active_fn_conv relu&

# various hidden_size_conv(tanh)
# CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H5-tanh --mode CNN --num_layer_conv 3 --hidden_size_conv 5 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H10-tanh --mode CNN --num_layer_conv 3 --hidden_size_conv 10 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H20-tanh --mode CNN --num_layer_conv 3 --hidden_size_conv 20 --active_fn_conv tanh&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name CNN-L3-H30-tanh --mode CNN --num_layer_conv 3 --hidden_size_conv 30 --active_fn_conv tanh&
