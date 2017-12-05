# training. Tune number of layers in D
python train.py --dataroot datasets/2russ2qi --display_id 1 --name 2russ2qi_D5 --model cycle_gan --checkpoints_dir checkpoints/ --save_epoch_freq 20 --n_layers_D 5

# testing
#python test.py --dataroot datasets/2russ2qi --name 2russ2qi_D5 --model cycle_gan --phase test
