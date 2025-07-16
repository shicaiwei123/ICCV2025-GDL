import os

train_audio_dir = '/home/ssd2/kinect_sound/audio/train'
test_audio_dir = '/home/ssd2/kinect_sound/audio/test'


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


test_video_list = get_file_list('/home/ssd2/kinect_sound/visual/val')

for i, item in enumerate(test_video_list):
    if i % 500 == 0:
        print('*******************************************')
        print('{}/{}'.format(i, len(test_video_list)))
        print('*******************************************')
    mp4_filename = item
    wav_path=os.path.join(test_audio_dir, item.split('/')[-2])
    if not  os.path.exists(wav_path):
        os.makedirs(wav_path)
    wav_filename = os.path.join(wav_path, item.split('/')[-1].split('.')[0] + '.wav')
    print(mp4_filename)
    print(wav_filename)
    if os.path.exists(wav_filename):
        pass
    else:
        os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))

# train set processing
# test_video_list = get_file_list('/home/ssd2/kinect_sound/visual/train')
#
# for i, item in enumerate(test_video_list):
#     if i % 500 == 0:
#         print('*******************************************')
#         print('{}/{}'.format(i, len(test_video_list)))
#         print('*******************************************')
#     mp4_filename = item
#     wav_path=os.path.join(train_audio_dir, item.split('/')[-2])
#     if not  os.path.exists(wav_path):
#         os.makedirs(wav_path)
#     wav_filename = os.path.join(wav_path, item.split('/')[-1].split('.')[0] + '.wav')
#     print(mp4_filename)
#     print(wav_filename)
#     if os.path.exists(wav_filename):
#         pass
#     else:
#         os.system('ffmpeg -i {} -acodec pcm_s16le -ar 16000 {}'.format(mp4_filename, wav_filename))
