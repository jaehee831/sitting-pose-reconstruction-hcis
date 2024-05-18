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
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
from collections import Counter

walk_types = {
    0: 'forward',
    1: 'back',
}

# connectedSock = bindUE4()


def angle_speed_estimation(ori_data, hmdYaw):

    ori_data = np.array(ori_data)
    low_image = np.min(ori_data, axis=0)
    high_image = np.max(ori_data, axis=0)
    normal_image = np.mean(ori_data, axis= 0)
    point_diff = high_image - low_image
    point_diff = np.max(point_diff) / point_diff

    input_image = (ori_data - normal_image)* point_diff / 200

    # foot crop
    foot_size = 32
    # find center of foot
    foot_signal_lower_bound = ori_data.mean() + 3.0 * ori_data.std() #over 99.997%
    #foot_signal_lower_bound = ori_data.min(axis=0) + 6.5 * ori_data.std() #over 99.997%
    print("foot signal bound:", foot_signal_lower_bound)
    # foot_signal_lower_bound = 585


    foot_x = np.where(ori_data >= foot_signal_lower_bound)[1].astype(int)
    foot_y = np.where(ori_data >= foot_signal_lower_bound)[2].astype(int)
    foot_cordination = list(zip(foot_x, foot_y))
    print("len_cordination {}".format(len(foot_cordination)))
    if len(foot_cordination)<=1:
        speed=0
        angle=0
        return speed, angle

    foot_cluster = KMeans(n_clusters=2, random_state=0).fit(foot_cordination)
    centers_of_cluster = foot_cluster.cluster_centers_

    slope = ((64-centers_of_cluster[0][0])-(64-centers_of_cluster[1][0])) / (centers_of_cluster[0][1]-centers_of_cluster[1][1])
    orthogonal_slope = -1/slope
    angle_a = np.degrees(np.arctan(orthogonal_slope))
    angle_a = angle_a - 90
    angle_a = angle_a * (-1)
    angle_b = angle_a - 180


    ## track foot speed
    foot_track_start_point = -1
    foot_cluster_buffer = -1

    for i in range(len(ori_data[:])):
        max_pressure_index = np.unravel_index(ori_data[14-i].argmax(), ori_data[14-i].shape)
        foot_cluster_num = foot_cluster.predict([list(max_pressure_index)])
        if i == 0:
            foot_cluster_buffer = foot_cluster_num
        if foot_cluster_buffer != foot_cluster_num:
            if foot_track_start_point == -1:
                foot_track_start_point = i
                foot_cluster_buffer = foot_cluster_num
            else:
                foot_interval = i - foot_track_start_point
                break
        if i == 14:
            foot_interval = 1000000

    speed_est = (17/foot_interval) * 60 ## (Frame per second/step per frame) * 60 second = steps per min
    
    # _, _, _, _, hmdYaw = DogChasingEnv.get_state()
    # hmdYaw = 0

    if hmdYaw >= 0:
        if abs(angle_a - hmdYaw) <= 90 :
            angle_est = angle_a
        else:
            angle_est = angle_b
    else:
        if abs(angle_b - hmdYaw) <= 90:
            angle_est = angle_b
        else:
            angle_est = angle_a

    # SensorWalk(connectedSock, speed_est*5, 1, 0, angle_a)

    # print("Speed_est: " + speed_est)
    # print("Angle_est: " + angle_est)
    print(hmdYaw)
    print("angle: ", angle_est)
    print("speed: ", speed_est)
    print(orthogonal_slope)
    print("cluster center: ", centers_of_cluster)
    
    return speed_est, angle_est

class DogChasingEnv:
    def __init__(self):
        self.connect()

        '''
        SetVisibleCircle
        SetDogMaxWalkSpeed
        SetPlayerMaxWalkSpeedDogChasingEnv
        SetDogLocation
        SetPlayerLocation
        '''
        self.exit = mp.Event()
        self.angle_queue, self.speed_queue, self.state_queue = Manager().Queue(), Manager().Queue(), Manager().Queue()
        self.axisX_queue = Manager().Queue()
        self.process = mp.Process(target=self._move_command, args=())
        self.process.start()
        
    def connect(self):
        self.sock = bindUE4()
        
    def get_state(self):
        self.state_queue.put("sleep")

        dog_location = GetDogLocation(self.sock)
        # print(dog_location)
        player_location = GetPlayerLocation(self.sock)
        distance = GetDistance(self.sock)
        player_state = GetPlayerState(self.sock)
        hmdYaw = GetCameraYaw(self.sock)
        dog_location = [float(i) for i in dog_location]
        player_location = [float(i) for i in player_location]
        distance = float(distance)/100
        player_state = int(player_state)

        self.state_queue.put("wakeup")
        
        return dog_location, player_location, distance, player_state, hmdYaw
    
    def move(self, angle, speed, axisX):
        # self.angle = math.cos(angle), math.sin(angle)
        self.angle = angle
        self.speed = speed
        self.axisX = axisX
    
        self.speed_queue.put(self.speed)
        self.angle_queue.put(self.angle)
        self.axisX_queue.put(self.axisX)
    
    def _move_command(self):
        # speed, angle = 0, (0, 0)
        speed, angle, axisX = 0, 0, 0
        while not self.exit.is_set():
            if not self.speed_queue.empty():
                while not self.speed_queue.empty():
                    speed = self.speed_queue.get()

            if not self.angle_queue.empty():
                while not self.angle_queue.empty():
                    angle = self.angle_queue.get()
            if not self.axisX_queue.empty():
                while not self.axisX_queue.empty():
                    axisX = self.axisX_queue.get()
            # SensorWalk(self.sock, speed * 10, angle[0], angle[1])

            SensorWalk(self.sock, speed*5, axisX, 0, angle)
            if not self.state_queue.empty():
                if self.state_queue.get() == "sleep":
                    while not self.state_queue.get() == "wakeup":
                        time.sleep(0.01)

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
        input_buffer = input_buffer.astype(np.float32)self.s
    '''
    def preprocessing(self, data):
        data = (data - 500) / 200
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
        data = np.clip(data, 0.0, 1.0)
        return data
    def angle_speed_estimation(self, ori_data, hmdYaw):

        ori_data = np.array(ori_data)

        point_diff = np.max(self.total_range) / self.total_range

        ori_data = (ori_data - self.total_mu)* point_diff / 200

        # foot crop
        foot_size = 32
        # find center of foot
        foot_signal_lower_bound = ori_data.mean() + 4.0 * ori_data.std() #over 99.997%
        #foot_signal_lower_bound = ori_data.min(axis=0) + 6.5 * ori_data.std() #over 99.997%
        print("foot signal bound:", foot_signal_lower_bound)
        # foot_signal_lower_bound = 585


        foot_x = np.where(ori_data >= foot_signal_lower_bound)[1].astype(int)
        foot_y = np.where(ori_data >= foot_signal_lower_bound)[2].astype(int)
        foot_cordination = list(zip(foot_x, foot_y))
        print("len_cordination {}".format(len(foot_cordination)))
        if len(foot_cordination)<=1:
            speed=0
            angle=0
            return speed, angle

        foot_cluster = KMeans(n_clusters=2, random_state=0).fit(foot_cordination)
        centers_of_cluster = foot_cluster.cluster_centers_

        slope = ((64-centers_of_cluster[0][0])-(64-centers_of_cluster[1][0])) / (centers_of_cluster[0][1]-centers_of_cluster[1][1])
        orthogonal_slope = -1/slope
        angle_a = np.degrees(np.arctan(orthogonal_slope))
        angle_a = angle_a - 90
        angle_a = angle_a * (-1)
        angle_b = angle_a - 180


        ## track foot speed
        foot_track_start_point = -1
        foot_cluster_buffer = -1

        for i in range(len(ori_data[:])):
            max_pressure_index = np.unravel_index(ori_data[14-i].argmax(), ori_data[14-i].shape)
            foot_cluster_num = foot_cluster.predict([list(max_pressure_index)])
            if i == 0:
                foot_cluster_buffer = foot_cluster_num
            if foot_cluster_buffer != foot_cluster_num:
                if foot_track_start_point == -1:
                    foot_track_start_point = i
                    foot_cluster_buffer = foot_cluster_num
                else:
                    foot_interval = i - foot_track_start_point
                    break
            if i == 14:
                foot_interval = 1000000

        speed_est = (17/foot_interval) * 60 ## (Frame per second/step per frame) * 60 second = steps per min
        
        # _, _, _, _, hmdYaw = DogChasingEnv.get_state()
        # hmdYaw = 0

        if hmdYaw >= 0:
            if abs(angle_a - hmdYaw) <= 90 :
                angle_est = angle_a
            else:
                angle_est = angle_b
        else:
            if abs(angle_b - hmdYaw) <= 90:
                angle_est = angle_b
            else:
                angle_est = angle_a

        # SensorWalk(connectedSock, speed_est*5, 1, 0, angle_a)

        # print("Speed_est: " + speed_est)
        # print("Angle_est: " + angle_est)
        print(hmdYaw)
        print("angle: ", angle_est)
        print("speed: ", speed_est)
        print(orthogonal_slope)
        print("cluster center: ", centers_of_cluster)
        
        return speed_est, angle_est

    def predict(self, frame, hmdYaw):
        self.input_buffer += frame
        #self.input_buffer = self.input_buffer[-15:]
        print(len(self.input_buffer))
        if len(self.input_buffer) < 200:
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing(np.array(self.input_buffer))
        elif len(self.input_buffer) in range(200, 210): 
            self.mat = np.mean(self.input_buffer, axis=0)
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing(np.array(self.input_buffer))
        elif len(self.input_buffer) < 500:
            speed, angle, walk_type = 0, 0, 'None'
            self.input_frame = self.preprocessing_v3(np.array(self.input_buffer))
        elif len(self.input_buffer) >= 500:
            if len(self.input_buffer) in range(500, 510):
                total_image = self.input_buffer
                total_min, total_max = np.min(total_image, axis=0), np.max(total_image, axis=0)
                self.total_mu = np.mean(total_image, axis=0)
                self.total_range = total_max - total_min
            #'''
            input_frame = self.preprocessing_v3(np.array(self.input_buffer[-self.window_size:]))
            self.input_frame = input_frame
            input_frame = torch.Tensor(input_frame).cuda()
            input_frame = input_frame.unsqueeze(0)
            '''
            # use except denoising
            self.input_frame = np.array(self.input_buffer)
            input_frame = torch.Tensor(self.input_buffer)
            input_frame = input_frame.unsqueeze(0)
            input_frame = (input_frame-500)/200
            '''
            

            '''
            sensor_out, classifi_out = model(input_frame)#, self.device)
            speed, angle = sensor_out.cpu().detach().numpy()[0].tolist()
            speed = speed*190 + 50
            angle *= 180
            angle += 180
            classifi_out = list(classifi_out.cpu().detach().numpy()[0])
            walk_type_int = classifi_out.index(max(classifi_out))
            walk_type = walk_types[walk_type_int]
            '''
            #_, motion, _ = model(input_frame) # conv2d_v2
            motion = model(input_frame) # conv2d
            as_frame = np.array(np.array(self.input_buffer[-15:]))
            speed, angle = self.angle_speed_estimation(as_frame, hmdYaw)

            motion = list(motion.cpu().detach().numpy()[0])
            motion = motion.index(max(motion))
            walk_type = walk_types[motion]

        else:
            speed, angle, walk_type = 0, 0, 'None'

        return speed, angle, walk_type


def main(
        model,
        normalize = False
    ):

    # make envself.s
    # use when testing with unreal
    env = DogChasingEnv()
    # hmdYaw = 0.0
    # initialize sensors
    sensor = MultiSensors(normalize=normalize)
    print("initializing sensors...")
    sensor.init_sensors()
    #fps monitoring
    fpsMonitor = FramerateMonitor()
    i = 0
    steady = []
    while True:
        total_images = sensor.get_all()
        _, _, _, _, hmdYaw = env.get_state()
        speed, angle, walk_type = model.predict(total_images, float(hmdYaw))
        steady.append(walk_type)

        axisX = 0
        occurence_count = Counter(steady[-10:])
        forwardback = occurence_count.most_common(1)[0][0]
        # print(forwardback, steady[-10:])
        if forwardback == 'back':
            axisX = -1
        else:
            axisX = 1
        # use when testing with unreal
        env.move(angle, speed, axisX)

        #visualize
        if i > window_size:
            visual_image = copy.deepcopy(model.input_frame[-1])
            #visual_image = (visual_image - 500) / 200
            visual_image *= 255
            visual_image = visual_image.astype(np.uint8)
            visual_image = cv2.resize(visual_image, (500, 500))
            # cv2.imshow("Pressure", visual_image)
            if cv2.waitKey(1) & 0xff == 27:
                break
        

        #fps
        fpsMonitor.tick()
        main_fps = round(fpsMonitor.getFps())
        sensor_fps = sensor.getFps()

        #verbose
        # print(f"FPS : {sensor_fps}, FPS2: {main_fps}, Speed: {speed:5>3.2f}, Angle: {angle:5>3.2f}, Types: {walk_type}")
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

    saved_path = "./models/hc/model__conv_2d_lr5e-05_w_size15_302384/model_60.pth"
    model = nn.DataParallel(vanilla_conv2d_v2(window_size))
    model.load_state_dict(torch.load(saved_path))

    model.module.state_dict()
    

    #wrap model
    main_model = model_wrapper(model.eval(), window_size=window_size)
    main(main_model)

    '''
    env = DogChasingEnv()
    while True:
        time.sleep(1)
        env.move(0, 300)
    env.close()
    '''
