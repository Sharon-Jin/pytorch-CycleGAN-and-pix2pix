# sudo pip install visdom
# sudo pip install dominate

# start visdom server to monitor the training process: 
# sudo python -m visdom.server

# training
python train.py --dataroot datasets/ukiyoe2photo --display_id 1 --name ukiyoe2photo --model cycle_gan --checkpoints_dir checkpoints/
