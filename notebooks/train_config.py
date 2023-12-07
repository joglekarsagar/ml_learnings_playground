# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = 'gpt_train_string_reversal'
# wandb_run_name=''

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 16
block_size = 64
gradient_accumulation_steps = 5 * 8


# this makes total number of tokens be 300B
max_iters = 50000
lr_decay_iters = 40000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 500

# weight decay
weight_decay = 1e-1

# Model stuff
n_embd = 512
n_head = 4
n_layer = 4

learning_rate = 1e-4 # max learning rate
min_lr = 1e-6 # learning_rate / 10 usually

max_iters = 600000 # total number of training iterations

beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla


bias=False  
dropout=0.1

out_dir = "./checkpoints/"