# 5. Best FCN & CNN

# L5-H1024-relu-b128-e500-adam(FCN)
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H256-relu-lr0005 --mode FCN --num_layer 3 --hidden_size 256 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H512-relu-lr0005 --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H1024-relu-lr0005 --mode FCN --num_layer 3 --hidden_size 1024 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&

CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L5-H256-relu-lr0005 --mode FCN --num_layer 5 --hidden_size 256 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L5-H512-relu-lr0005 --mode FCN --num_layer 5 --hidden_size 512 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
# CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-best-lr0005 --mode FCN --num_layer 5 --hidden_size 1024 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.0005&

CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H256-relu-lr001 --mode FCN --num_layer 3 --hidden_size 256 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H512-relu-lr001 --mode FCN --num_layer 3 --hidden_size 512 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L3-H1024-relu-lr001 --mode FCN --num_layer 3 --hidden_size 1024 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&

CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L5-H256-relu-lr001 --mode FCN --num_layer 5 --hidden_size 256 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-L5-H512-relu-lr001 --mode FCN --num_layer 5 --hidden_size 512 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&
# CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name fcn-best-lr001 --mode FCN --num_layer 5 --hidden_size 1024 --active_fn relu --batch_size 128 --num-epochs 100 --learning_rate 0.001&


# L3-H30-tanh-b128-e500-adam(CNN) + L5-H1024-relu-b128-e500-adam(FCN)
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name cnn-L3-H30-tanh-lr0005 --mode CNN --num_layer 5 --hidden_size 1024 --active_fn relu \
--num_layer_conv 3 --hidden_size_conv 30 --active_fn_conv tanh --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name cnn-L3-H20-tanh-lr0005 --mode CNN --num_layer 5 --hidden_size 1024 --active_fn relu \
--num_layer_conv 3 --hidden_size_conv 20 --active_fn_conv tanh --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
CUDA_VISIBLE_DEVICES=1 python pa_1.py --exp-name cnn-L3-H10-tanh-lr0005 --mode CNN --num_layer 5 --hidden_size 1024 --active_fn relu \
--num_layer_conv 3 --hidden_size_conv 10 --active_fn_conv tanh --batch_size 128 --num-epochs 100 --learning_rate 0.0005&
