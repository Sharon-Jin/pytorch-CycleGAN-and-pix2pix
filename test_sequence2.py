# Note that ".mkv" file can be played using VLC under mac os

import cv2, os, sys, pdb, shutil
import torch

# configurations
test_video_path = 'datasets/videos/test2.mp4'
experiment_name = "3russ2jin"
extract_folder = 'datasets/' + experiment_name
epoch = 200
current_dir = os.getcwd()
#


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    '''
    current_dir = os.getcwd()
    extract_folder = os.path.join(os.getcwd(), test_video_path.replace('.mp4', ''))
    mkdir_if_not_exist(extract_folder)

    # resize video
    resize_video_path = test_video_path.replace('.MOV', '_resized.mp4')
    resize_video_command = "ffmpeg -i " + test_video_path + " -filter:v \"crop=1080:1080:420:0\" -c:a copy " + resize_video_path
    os.system(resize_video_command)

    # extract each frame of the video
    extract_folder_testA = os.path.join(extract_folder, 'testA')
    mkdir_if_not_exist(extract_folder_testA)
    extract_folder_testB = os.path.join(extract_folder, 'testB')
    mkdir_if_not_exists(extract_folder_testB)
    copy_command = "cp %s/* %s/" % (extract_folder_testA, extract_folder_testB)
    os.system(copy_command)
    extract_video_command = "ffmpeg -i " + resize_video_path + " " + extract_folder_testA + "/%03d.jpg"
    os.system(extract_video_command)
    '''
    
    # extract audio
    audio_path = test_video_path.replace("mp4", "mp3")
    extract_audio_command = "ffmpeg -i " + test_video_path + " -q:a 0 -map a " + audio_path
    os.system(extract_audio_command)

    # forward all the images
    run_pytorch_command = ('python test.py --which_epoch %d --dataroot %s --name %s --model cycle_gan --phase test --serial_batches --which_direction BtoA' % (epoch, extract_folder, experiment_name))
    os.system(run_pytorch_command)
    
    fake_folder = extract_folder + "_fake"
    mkdir_if_not_exist(fake_folder)
    # copy all the files from original result folder to _fake folder
    copy_result_command = ("cp results/%s/test_%d/images/* %s" % (experiment_name, epoch, fake_folder))
    os.system(copy_result_command)

    extracted_files = [s for s in os.listdir(fake_folder) if s.endswith('_fake_A.png')]
    extract_files_num = len(extracted_files)

    # combine all output images to get a full video
    output_video_no_sound_path = test_video_path.replace('.mp4', '_no_sound.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(os.path.join(current_dir, output_video_no_sound_path),fourcc, 30.0, (256, 256), True)

    for i in range(extract_files_num):
        fake_frame_name = os.path.join(fake_folder, ("%03d_fake_A.png" % (i+1)))
        if os.path.exists(fake_frame_name):
            img = cv2.imread(fake_frame_name)
            out.write(img)
            print("writing %s" % fake_frame_name)
        else:
            print("path %s not exist!" % fake_frame_name)

    # Release everything if job is finished
    out.release()
    print("Finished getting fake video (without sound)")

    # add audio to video
    output_video_path = test_video_path.replace('.mp4', '_output.mkv')
    add_audio_command = "ffmpeg -i " + output_video_no_sound_path + " -i " + audio_path + " -map 0 -map 1 -codec copy " + output_video_path
    os.system(add_audio_command)

if __name__ == "__main__":
    main()
