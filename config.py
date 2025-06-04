embd_dim = 768

max_seq_len = 256

batch_size = 2048
mini_batch_size = 128
grad_acc_steps = batch_size // mini_batch_size
start_lr = 1e-4
min_lr = 1e-5
lr_decay_steps = 1000000 ### decrease from start_lr to min_lr in lr_decay_steps steps

weight_decay = 1e-1