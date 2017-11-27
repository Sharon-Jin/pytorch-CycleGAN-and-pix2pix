# sudo pip install visdom
# sudo pip install dominate

# start visdom server to monitor the training process: 
# sudo python -m visdom.server

# training
#python train.py --dataroot datasets/3russ2jin --display_id 1 --name 3russ2jin --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 40
python train.py --dataroot datasets/3russ2jin --display_id 1 --name 3russ2jin_mask --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 20 --continue_train --which_epoch 160 --face_mask

# please refer to options/train_options.py for more training configs
