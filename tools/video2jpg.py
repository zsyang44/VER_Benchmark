from __future__ import print_function, division
import os
import sys
import subprocess


def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)

    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_dir_path):
        os.mkdir(dst_class_path)
    for file_name in os.listdir(class_path):
        if '.mp4' not in file_name:
            continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)
        video_file_path = os.path.join(class_path, file_name)
        try:
            if os.path.exists(dst_directory_path):
                if not os.path.exists(os.path.join(dst_directory_path, 'image00001.jpg')):
                    subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                    print('remove {}'.format(dst_directory_path))
                    os.makedirs(dst_directory_path)
                else:
                    continue
            else:
                os.mkdir(dst_directory_path)
        except:
            print(dst_directory_path)
            continue

        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/%06d.jpg\"'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')


if __name__ == "__main__":
    dir_path = "D:\courses\Emotion_detection\Trial\Dataset\VideoEmotion8"  # avi directory
    dst_dir_path = "D:\courses\Emotion_detection\Trial\Dataset\VideoEmotion8--img"  # jpg directory

    class_name = "Sadness"
    class_process(dir_path, dst_dir_path, class_name)
    # class_list = ["Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"]
    #
    # for class_name in class_list:#os.listdir(dir_path)
    #     class_process(dir_path, dst_dir_path, class_name)
