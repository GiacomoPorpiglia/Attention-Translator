embd_dim = 512

max_seq_len = 128

batch_size = 1024
mini_batch_size = 64
bucket_size = mini_batch_size*20
grad_acc_steps = batch_size // mini_batch_size
start_lr = 1e-4
min_lr = 1e-5
lr_decay_steps = 100000 ### decrease from start_lr to min_lr in lr_decay_steps steps

weight_decay = 1e-1