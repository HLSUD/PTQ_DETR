# ----------Quantization config----------
w_bit = 8
a_bit = 8

metric="hessian"
search_round = 3

# qconv args
conv_eq_alpha = 0.01
conv_eq_beta = 1.2
conv_eq_n = 100
conv_n_V = 1
conv_n_H = 1

# qlinear args
linear_eq_alpha = 0.01
linear_eq_beta = 1.2
linear_eq_n = 100
linear_n_V = 1
linear_n_H = 1
linear_n_a=1
bias_correction=True

# attention/matmul args
attn_eq_alpha=0.01
attn_eq_beta=1.2
attn_eq_n=100
attn_n_V_A = 1
attn_n_H_A = 1
attn_n_G_A = 1
attn_n_V_B = 1
attn_n_H_B = 1
attn_n_G_B = 1

# ----------Model config----------
model_name = "facebook/detr-resnet-50"

# ----------Dataset config----------
dataset_dir = 'path_to_dataset'
annotation_file = 'path_to_ann_files'
batch_size = 1
num_workers = 1
calib_size = 32