import sys; sys.path.insert(0, '.')
from sensors.Sensor_new import MultiSensors
from models.hc_model import vanilla_conv2d_v2
#from models.sh_model import vanilla_conv2d_v2
from models.sh_const import ANGLE_CLASSES_NUM, CLASS_NUM, REGRESS_NUM
import torch
import os
import copy
import cv2
from scipy import ndimage
import numpy as np
from app.FramerateMonitor import FramerateMonitor
from TcpSender import *
import math
import random
import multiprocessing as mp
from multiprocessing import Manager
from models.lstm_model import LSTMCNN
from scipy.ndimage import gaussian_filter
import h5py
walk_types = {
    0: "forward",
    1: "back",
}
def remove_zero(data):
    idxs = []
    for i in range(len(data)):
        if data[i].mean() == 0.0:
            idxs.append(i)
    data = np.delete(data, idxs, 0)
    return data

NORMALIZE_PATH = './models/hc/norm_matrix.hdf5'

class DogChasingEnv:
    def __init__(self):
        self.connect()

        '''
        SetVisibleCircle
        SetDogMaxWalkSpeed
        SetPlayerMaxWalkSpeed
        SetPlayerMaxAccelation

        SetDogLocation
        SetPlayerLocation
        '''
        self.exit = mp.Event()
        self.angle_queue, self.speed_queue = Manager().Queue(), Manager().Queue()
        self.process = mp.Process(target=self._move_command, args=())
        self.process.start()
        
    def connect(self):
        self.sock = bindUE4()
        
    def get_state(self):
        dog_location = GetDogLocation(self.sock)
        player_location = GetPlayerLocation(self.sock)
        distance = GetDistance(self.sock)
        player_state = GetPlayerState(self.sock)
        
        dog_location = [float(i) for i in dog_location]
        player_location = [float(i) for i in player_location]
        distance = float(distance)/100
        player_state = int(player_state)
        return dog_location, player_location, distance, player_state
    
    def move(self, angle, speed):
        self.angle = math.cos(angle), math.sin(angle)
        self.speed = speed
    
        self.speed_queue.put(self.speed)
        self.angle_queue.put(self.angle)
    
    def _move_command(self):
        speed, angle = 0, (0, 0)
        while not self.exit.is_set():
            if not self.speed_queue.empty():
                while not self.speed_queue.empty():
                    speed = self.speed_queue.get()

            if not self.angle_queue.empty():
                while not self.angle_queue.empty():
                    angle = self.angle_queue.get()
            SensorWalk(self.sock, speed, angle[0], angle[1])

    def close(self):
        self.exit.set()
        self.process.join()



class model_wrapper:
    def __init__(self, model, window_size=20, device='cuda:0'):
        self.model = model
        self.device = device
        self.input_buffer = []
        self.window_size = window_size
    '''
    def preprocessing(self, input_buffer):
        input_buffer = input_buffer.astype(np.uint8)
        for i in range(len(input_buffer)):
            input_buffer[i] = cv2.fastNlMeansDenoising(input_buffer[i])
        input_buffer = ndimage.median_filter(input_buffer, 2)
        input_buffer = input_buffer.astype(np.float32)
        input_buffer -= input_buffer.mean()
        input_buffer /= input_buffer.std()
        input_buffer *= 0.1
        input_buffer += 0.5
        return input_buffer
    '''

    def preprocessing(self, data):
        # data = (data - 500) / 200
        return data

    def preprocessing_v2(self, data):
        data = data.astype(np.float32)
        data -= data.mean()
        data /= data.std()
        data *= 0.1
        data += 0.5

        return data

    def preprocessing_v3(self, data):
        data = (data - self.mat) / 200
        # data = np.clip(data, 0.0, 1.0)
        data = data + 0.5
        return data


    def predict(self, frame, i):
        self.input_buffer += frame
        #self.input_buffer = self.input_buffer[-15:]
        # 500 frame for initialize, 200 frame for buffer, after that test start
        if len(self.input_buffer) < 500:
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing(np.array(self.input_buffer))
        elif len(self.input_buffer) == 500: 
            self.mat = np.mean(self.input_buffer, axis=0)
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing(np.array(self.input_buffer))
        elif len(self.input_buffer) < 700:
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing_v3(np.array(self.input_buffer))
        elif len(self.input_buffer) >= 700:
            
            #'''
            input_frame = self.preprocessing_v2(np.array(self.input_buffer[-self.window_size:]))
            self.input_frame = input_frame
            input_frame = torch.Tensor(input_frame).cuda()
            input_frame = input_frame.unsqueeze(0)

            #_, motion, _ = model(input_frame) # conv2d_v2
            motion = model(input_frame) # conv2d

            motion = list(motion.cpu().detach().numpy()[0])
            motion = motion.index(max(motion))
            walk_type = walk_types[motion]

            speed =0
            angle =0


        else:
            speed, angle, walk_type = 0, 0, 'None'
        NORMALIZE_PATH
        return speed, angle, walk_type


def main(
        model,
        normalize = False
    ):

    # make env
    # use when testing with unreal
    #env = DogChasingEnv()

    # initialize sensors
    sensor = MultiSensors(normalize=normalize)
    print("initializing sensors...")
    sensor.init_sensors()

    #fps monitoring
    fpsMonitor = FramerateMonitor()
    i = 0
    while True:
        print(len(model.input_buffer))
        total_images = sensor.get_all()

        speed, angle, walk_type = model.predict(total_images, i)

        # use when testing with unreal
        #env.move(angle, speed)

        #visualize
        if i > window_size:
            visual_image = copy.deepcopy(model.input_frame[-1])
            #visual_image = (visual_image - 500) / 200
            visual_image *= 255
            visual_image = visual_image.astype(np.uint8)
            visual_image = cv2.resize(visual_image, (500, 500))
            cv2.imshow("Pressure", visual_image)
            if cv2.waitKey(1) & 0xff == 27:
                break
        

        #fps
        fpsMonitor.tick()
        main_fps = round(fpsMonitor.getFps())
        sensor_fps = sensor.getFps()

        #verbose
        print(f"FPS : {sensor_fps}, FPS2: {main_fps}, Speed: {speed:5>3.2f}, Angle: {angle:5>3.2f}, Types: {walk_type}")
        i += 1
    
    sensor.close()


if __name__ == "__main__":
    import torch.nn as nn
    
    
    # load model
    '''
    #saved_path= "./models/vanilla/singlePeople_0.0001_10_best.path.tar"
    saved_path= "./models/cropped/singlePeople_0.0001_10_best.path.tar"
    model = tile2openpose_conv3d(10)
    checkpoint = torch.load(saved_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    '''

    window_size = 15

    saved_path ="./models/hc/model__conv_2d_lr5e-05_w_size15_302384/model_60.pth"
    '''model__conv_2d_lr5e-05_w_size20_081066_hc''' 
    #model = nn.DataParallel(LSTMCNN(window_size))
    model = nn.DataParallel(vanilla_conv2d_v2(window_size))
    model.load_state_dict(torch.load(saved_path))

    model.module.state_dict()
    model.eval()

    #wrap model
    main_model = model_wrapper(model, window_size=window_size)
    main(main_model)

    '''
    env = DogChasingEnv()
    while True:
        time.sleep(1)
        env.move(0, 300)
    env.close()
    '''
    
