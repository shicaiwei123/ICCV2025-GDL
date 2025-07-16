import sys

import pandas as pd
import cv2
import os
import pdb


def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list


class videoReader(object):
    def __init__(self, video_path, frame_interval=1, frame_kept_per_second=1):
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.frame_kept_per_second = frame_kept_per_second

        #pdb.set_trace()
        self.vid = cv2.VideoCapture(self.video_path)
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.video_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_len = int(self.video_frames/self.fps)


    def video2frame(self, frame_save_path):
        self.frame_save_path = frame_save_path
        success, image = self.vid.read()
        count = 0
        while success:
            count +=1
            if count % self.frame_interval == 0:
                save_name = '{}/frame_{}_{}.jpg'.format(self.frame_save_path, int(count/self.fps), count)  # filename_second_index
                cv2.imencode('.jpg', image)[1].tofile(save_name)
            success, image = self.vid.read()


    def video2frame_update(self, frame_save_path):
        self.frame_save_path = frame_save_path

        count = 0
        frame_interval = int(self.fps/self.frame_kept_per_second)
        while(count < self.video_frames):
            ret, image = self.vid.read()
            if not ret:
                break
            if count % self.fps == 0:
                frame_id = 0
            if frame_id<frame_interval*self.frame_kept_per_second and frame_id%frame_interval == 0:
                save_name = '{0}/{1:05d}.jpg'.format(self.frame_save_path, count)
                cv2.imencode('.jpg', image)[1].tofile(save_name)

            frame_id += 1
            count += 1


class VGGSound_dataset(object):
    def __init__(self, path_to_dataset = '/home/ssd2/kinect_sound', frame_interval=1, frame_kept_per_second=1):
        self.path_to_video = os.path.join(path_to_dataset, 'val')
        self.frame_kept_per_second = frame_kept_per_second
        self.path_to_save = os.path.join(path_to_dataset, 'val_img', 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)


        self.video_list=get_file_list(self.path_to_video)

    def extractImage(self):

        for i, each_video in enumerate(self.video_list):
            each_video_path=each_video
            if i % 100 == 0:
                print('*******************************************')
                print('Processing: {}/{}'.format(i, len(self.video_list)))
                print('*******************************************')
            video_dir = each_video_path
            try:
                self.videoReader = videoReader(video_path=video_dir, frame_kept_per_second=self.frame_kept_per_second)

                save_dir = os.path.join(self.path_to_save, each_video_path.split('/')[-2],each_video_path.split('/')[-1].split('.')[0])
                # print(save_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    self.videoReader.video2frame_update(frame_save_path=save_dir)
                else:
                    continue
            except:
                print('Fail @ {}'.format(each_video))
                print(video_dir)
                # sys.exit(1)


vggsound = VGGSound_dataset()
vggsound.extractImage()