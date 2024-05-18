import time

import h5py
import os
import pandas as pd
import numpy as np
import pickle
import torch
import copy
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import matplotlib.pylab as plt

import cv2
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import random
from const import CLASS_TYPE_INDEX, INDEX_CLASS_TYPE, EXCEPT_NAMES, ANGLE_TO_CLASS

def remove_zero(data):
    idxs = []
    for i in range(len(data)):
        if data[i].mean() == 0.0:
            idxs.append(i)
    data = np.delete(data, idxs, 0)
    return data

def load_all_sequences(path):
    results = []
    for filename in os.listdir(path):

        if any([except_name in filename for except_name in EXCEPT_NAMES]):
            continue

        with h5py.File(path + filename, "r") as f:
            data = np.array(list(f["pressure"]))
            data = remove_zero(data)
            length = len(data)

            class_label, speed = filename.replace(".hdf5", "").split("_")
            speed = int(speed)
            if class_label == 'stand':
                speed = 0
            class_label = CLASS_TYPE_INDEX[class_label]

            speed = np.full((length, 1), speed)
            angle = np.full((length, 1), 0)
            class_label = np.full((length, 1), class_label)

        result = {
            "data": data,
            "speed": speed,
            "angle": angle,
            "class_label": class_label
        }
        results.append(result)
    return results
 
def rotate_sequence(data, angle):
    angle_value = random.randint(0, 360)
    for i in range(len(data)):
        angle[i] = (angle[i]+angle_value)%360
        aug_data1 = rotate(data[i], angle=angle_value, reshape=False, mode='reflect')
        data[i] = aug_data1
    return data, angle

def rotate_move(_origin_image, angle, x, y, foot_size):
    origin_image = _origin_image.copy()
    length = len(origin_image)

    foot_start = int((length - foot_size)/2)
    foot_img = origin_image[foot_start:foot_start+foot_size, foot_start:foot_start+foot_size].copy()

    other_idxs = list(set(range(length)) - set(range(foot_start, foot_start+foot_size)))
    other_side = origin_image[other_idxs, other_idxs]

    origin_image[foot_start:foot_start+foot_size, foot_start:foot_start+foot_size] = \
        np.random.normal(other_side.mean(), other_side.std(), foot_img.shape)

    foot_img = rotate(foot_img, angle=angle, reshape=False, mode='reflect')
    origin_image[x:x+foot_size, y:y+foot_size] = foot_img
    return origin_image

def rotate_move_sequence(data, angle):
    foot_size = 32

    angle_value = random.randint(0, 360)
    x, y = random.randint(0, len(data[0])-1 - foot_size), random.randint(0, len(data[0])-1 - foot_size)
    for i in range(len(data)):
        angle[i] = (angle[i]+angle_value)%360
        aug_data1 = rotate_move(data[i], angle_value, x, y, foot_size)
        data[i] = aug_data1
    return data, angle

def sliding_windows(array, window_size):
    total_data = []
    for i in range(len(array) - window_size):
        total_data.append([array[i:i+window_size]])
    return np.concatenate(total_data)

def show_pressure(img):
    ax = sns.heatmap(img, linewidth=0.5)
    plt.show()

def zoom_pressure(image, scale):
    assert image.shape[0] == image.shape[1] and len(image.shape) == 2
    origin_len = len(image)
    resized_image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_len = len(resized_image)
    diff = int(abs(resized_len - origin_len) / 2)
    if scale > 1:
        result = resized_image[diff:diff+origin_len, diff:diff+origin_len]
    else:
        result = image.copy()
        result[diff:diff+resized_len, diff:diff+resized_len] = resized_image

    return result

class PressureDataset(Dataset):
    def __init__(self, path, window_size):
        if type(path) == str:
            path = [path]
        elif type(path) == list:
            pass
        else:
            raise ValueError
        self.origin_sequences = []
        for pth in path:
            self.origin_sequences += load_all_sequences(pth)
            print(f"Loaded {pth}")

        self.window_size = window_size
        self.input_sequences = copy.deepcopy(self.origin_sequences)

        self.seq_indexs = []
        start = 0
        for i, seq in enumerate(self.input_sequences):
            end = start + len(seq["speed"]) - (self.window_size - 1)
            self.seq_indexs.append((i, start, end))
            start = end

    def __len__(self):
        return self.seq_indexs[-1][-1]

    def __getitem__(self, idx):
        for i, start, end in self.seq_indexs:
            if idx >= start and idx < end:
                real_idx = idx - start

                data = self.input_sequences[i]["data"][real_idx:real_idx+self.window_size]
                speed = self.input_sequences[i]["speed"][real_idx+self.window_size-1]
                angle = self.input_sequences[i]["angle"][real_idx+self.window_size-1]
                class_label = self.input_sequences[i]["class_label"][real_idx+self.window_size-1]
                break
        return self.preprocessing_v2(data, speed, angle, class_label)

    def preprocessing(self, data, speed, angle, class_label):
        data = (data - 500) / 200
        speed = (speed - 50) / 190
        angle = (angle - 180) / 180
        value_label = np.concatenate((speed, angle), axis=0)
        return torch.FloatTensor(data), torch.FloatTensor(value_label), torch.LongTensor(class_label)

    def preprocessing_v2(self, data, speed, angle, class_label):
        data = data.astype(np.float32)
        data -= data.mean()
        data /= data.std()
        data *= 0.1
        data += 0.5

        speed = (speed - 50) / 190
        angle = ANGLE_TO_CLASS[angle[0]]
        return torch.FloatTensor(data), torch.FloatTensor(speed), torch.LongTensor(class_label), torch.LongTensor([angle])

    def augment(self):
        self.input_sequences = copy.deepcopy(self.origin_sequences)

        for i in range(len(self.input_sequences)):

            data = self.input_sequences[i]["data"]
            angle = self.input_sequences[i]["angle"]

            # rotate and move
            random_val = random.randint(0, 1)
            if random_val == 0:
                data, angle = rotate_sequence(data, angle)
            else:
                data, angle = rotate_move_sequence(data, angle)

            # zoom
            scale = random.uniform(0.7, 1.3)
            for j in range(len(data)):
                data[j] = zoom_pressure(data[j], scale)

            # blur
            '''
            if random.randint(0, 1) == 0:
                blur_value = random.uniform(0.1, 1)
                for j in range(len(data)):
                    data[j] = gaussian_filter(data[j], blur_value)
            '''

            # gaussian noises
            '''
            if random.randint(0, 1) == 0:
                np.add(
                    data,
                    np.random.normal(
                        random.randint(-10, 10), random.randint(3, 8),
                        data.shape
                    ),
                    out=data,
                    casting="unsafe"
                )
            '''
            data = data.astype(np.uint8)
            for j in range(len(data)):
                data[j] = cv2.fastNlMeansDenoising(data[j])
            data = ndimage.median_filter(data, 2)

            self.input_sequences[i]["data"] = data
            self.input_sequences[i]["angle"] = angle


if __name__ == "__main__":
    path = "./data/cyh/"
    window_size = 40
    loader = DataLoader(PressureDataset(path, window_size), batch_size=1024, shuffle=True)
    for x, y, y2 in loader:
        print(f"speed: {y[0][0]}, angle: {y[0][1]}, class: {INDEX_CLASS_TYPE[y2[0].item()]}")
        visual_all = x[0].cpu().detach().numpy()
        '''
        min_val = 0.01
        max_val = 0.03
        visual_all = (visual_all - min_val)/(max_val - min_val)
        '''

        visual_all *= 255

        visual_all = visual_all.astype(np.uint8)

        for visual in visual_all:

            visual = zoom_pressure(visual, 0.7)

            visual = cv2.resize(visual, (500, 500), interpolation=cv2.INTER_AREA)

            cv2.imshow('image', visual)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
    cv2.destroyAllWindows()
    
    



