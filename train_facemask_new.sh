# sudo pip install visdom
# sudo pip install dominate

# start visdom server to monitor the training process: 
# sudo python -m visdom.server

# training
#python train.py --dataroot datasets/3russ2jin --display_id 1 --name 3russ2jin --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 40
#python train.py --dataroot datasets/2russ2qi --display_id 1 --name 2russ2qi_dlibmask --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 10 --face_mask --face_weight 2.0

python train.py --dataroot datasets/3russ2jin --display_id 1 --name 3russ2jin_dlibmask --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 5 --face_mask --face_weight 2.0 --continue_train --which_epoch 160

# please refer to options/train_options.py for more training configs
