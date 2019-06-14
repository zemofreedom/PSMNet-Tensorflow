# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:47:53 2019

@author: Administrator
"""

import os


def generate_image_list(data_dir='../SceneFlow/frames_cleanpass/', label_dir='../SceneFlow/disparity/'):
    # sub_dirs_data looks like `['C', 'A', 'B']`
    sub_dirs_data = [f1 for f1 in os.listdir(data_dir)
                     if os.path.isdir(os.path.abspath(os.path.join(data_dir, f1)))]
    sub_dirs_labels = [f2 for f2 in os.listdir(label_dir)
                       if os.path.isdir(os.path.abspath(os.path.join(label_dir, f2)))]
    f_train = open('train.lst', 'w')

    assert len(sub_dirs_data) == len(sub_dirs_labels)

    for sub_dir in sub_dirs_data:
        # data_complete_dir looks like `frames_cleanpass/A`
        data_complete_dir = os.path.join(data_dir, sub_dir)
        label_complete_dir = os.path.join(label_dir, sub_dir)
        label_index = []
        label_files = []
        data_files = []

        # subdir_num_path looks like `0087`
	    # subdir_left_abs_path looks like `frames_finalpass/TRAIN/C/0087/left`
        subdir_label_abs_path = str(label_complete_dir) + '/left'

	    # file looks like `0007.pfm`
        for file in os.listdir(subdir_label_abs_path):
            assert(os.path.isfile(str(label_complete_dir) + '/right/' + str(file)))
            label_files.append(str(label_complete_dir) + '/left/' + str(file) + '\t' +
                               str(label_complete_dir) + '/right/' + str(file))

        label_files.sort()
        label_files_length = len(label_files)

        # subdir_num_path looks like `0087`
        #for subdir_num_path in os.listdir(data_complete_dir):

        #subdir_num_abs_path = os.path.abspath(os.path.join(data_complete_dir, subdir_num_path))

        subdir_left_abs_path = str(data_complete_dir) + '/left'

	    # file looks like `0007.png`
        for file in os.listdir(subdir_left_abs_path):
            assert(os.path.isfile(str(data_complete_dir) + '/right/' + str(file)))
            data_files.append(str(data_complete_dir) + '/left/' + str(file) + '\t' +
				  str(data_complete_dir) + '/right/' + str(file))

        data_files_length = len(data_files)
        print('data_files_length of folder ' + str(sub_dir) + ': ' + str(data_files_length))
        data_files.sort()

        # The number of labels and data must be the same
        assert label_files_length == data_files_length

        for data_file, label_file in zip(data_files, label_files):
            line = str(data_file) + '\t' + str(label_file) + '\n'
            f_train.write(line)

    f_train.close()
    print("Image list generation completed!")


if __name__ == '__main__':
    generate_image_list()