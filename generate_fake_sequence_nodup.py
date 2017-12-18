# Note that ".mkv" file can be played using VLC under mac os

# Please copy drive/707/Original Video/testX.MOV to datasets/test/ folder and run this 

# test1.MOV: qiye
# test2.MOV: xiaohan
# test3/4.MOV: shangxuan

# configurations
#

import cv2, os, sys, pdb, shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--model-name', type=str)
parser.add_argument('-epoch', type=str, default='200')
parser.add_argument('-i', '--video-id', type=int)
args = parser.parse_args()

test_video_path = 'datasets/test/test{}.MOV'.format(args.video_id)
experiment_name = args.model_name
epoch = args.epoch

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        #delete all in the path
        shutil.rmtree(path)
        os.makedirs(path)

def main():
    current_dir = os.getcwd()
    extract_folder = os.path.join(os.getcwd(), 'datasets/test/{}'.format(experiment_name))
    mkdir_if_not_exist(extract_folder)

    # resize video
            # self.A_mask.resize_(A_mask.size()).copy_(A_mask)
            # self.B_mask.resize_(B_mask.size()).copy_(B_mask)
    resize_video_path = test_video_path.replace('.MOV', '_resized.mp4')
    resize_video_command = "ffmpeg -i " + test_video_path + " -filter:v \"crop=1080:1080:420:0\" -c:a copy " + resize_video_path
    os.system(resize_video_command)

    # extract each frame of the video
    extract_folder_testA = os.path.join(extract_folder, 'testA')
    mkdir_if_not_exist(extract_folder_testA)
    extract_folder_testB = os.path.join(extract_folder, 'testB')
    mkdir_if_not_exist(extract_folder_testB)
    copy_command = "cp %s/* %s/" % (extract_folder_testA, extract_folder_testB)
    extract_video_command = "ffmpeg -i " + resize_video_path + " " + extract_folder_testA + "/%03d.png"
    os.system(extract_video_command)
    os.system(copy_command)

    # extract audio
    audio_path = resize_video_path.replace("mp4", "mp3")
    extract_audio_command = "ffmpeg -i " + resize_video_path + " -q:a 0 -map a " + audio_path
    os.system(extract_audio_command)

    # forward all the images
    run_pytorch_command = ('python test.py --gpu_ids 0 --which_epoch %s --dataroot %s --name %s --model cycle_gan --phase test --serial_batches --resize_or_crop resize_and_crop --which_direction BtoA' % (epoch, extract_folder, experiment_name))
    os.system(run_pytorch_command)

    fake_folder = extract_folder + "_fake"
    mkdir_if_not_exist(fake_folder)
    # copy all the files from original result folder to _fake folder
    copy_result_command = ("cp results/%s/test_%s/images/* %s" % (experiment_name, epoch, fake_folder))
    os.system(copy_result_command)

    extracted_files = [s for s in os.listdir(fake_folder) if s.endswith('_fake_A.png')]
    extract_files_num = len(extracted_files)

    # combine all output images to get a full video
    output_video_no_sound_path = test_video_path.replace('.MOV', '_no_sound.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(current_dir, output_video_no_sound_path),fourcc, 30.0, (512, 256), True)

    for i in range(extract_files_num):
        fake_frame_name = os.path.join(fake_folder, ("%03d_fake_A.png" % (i+1)))
        real_frame_name = os.path.join(extract_folder_testA, ("%03d.png" % (i+1)))
        if os.path.exists(fake_frame_name) and os.path.exists(real_frame_name):
            fake_img = cv2.imread(fake_frame_name)
            real_img = cv2.resize(cv2.imread(real_frame_name), (256, 256))
            #pdb.set_trace()
            img = np.concatenate((real_img, fake_img), axis=1)
            out.write(img)
            print("writing %s" % fake_frame_name)
        else:
            print("path %s not exist!" % fake_frame_name)

    # Release everything if job is finished
    out.release()
    print("Finished getting fake video (without sound)")

    # add audio to video
    output_video_path =  os.path.join(fake_folder, 'fake_video.mkv')
    add_audio_command = "ffmpeg -i " + output_video_no_sound_path + " -i " + audio_path + " -map 0 -map 1 -codec copy " + output_video_path
    os.system(add_audio_command)

    # remove duplicated test images
    os.system('rm -rf %s' % extract_folder)
    
    # remove images generated from test a
    os.system('rm {ff}/*real_A.png {ff}/*fake_B.png {ff}/*rec_A.png'.format(ff=fake_folder))

if __name__ == "__main__":
    main()
