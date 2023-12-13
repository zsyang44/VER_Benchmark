from __future__ import print_function, division
import os
import sys
import subprocess
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np


def preprocess(dir_path, train_path, val_path):
    video_files = []
    for class_name in os.listdir(dir_path):
        class_path = os.path.join(dir_path, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_name, file_name)
            video_files.append(file_path)
    train, val = train_test_split(video_files, test_size=1/3, random_state=42)

    with open(train_path, 'w+') as train_file:
        for t in train:
            train_file.write(str(t)+'\n')

    with open(val_path, 'w+') as val_file:
        for v in val:
            val_file.write(str(v)+'\n')




if __name__ == "__main__":
    dir_path = "D:\courses\Emotion_detection\Trial\Dataset\VideoEmotion8--img"
    train_path = "D:\courses\Emotion_detection\Trial\\tools\\annotations\\ve8\\trainlist01.txt"
    val_path= "D:\courses\Emotion_detection\Trial\\tools\\annotations\\ve8\\testlist01.txt"

    preprocess(dir_path, train_path, val_path)
