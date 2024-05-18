import sys; sys.path.insert(0, '.')
import serial
from serial import Serial
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Manager

import copy
import time
from common import dataset_tools
from app.FramerateMonitor import FramerateMonitor
from app.AppContext import AppContext
from storage.Storage import createStorage

class Sensor:
    def __init__(self, port, baudrate, timeout):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.queue = Manager().Queue()
        self.exit = mp.Event()
        self.process = mp.Process(target=self._read, args=(self.queue,))
        
    def start(self):
        self.process.start()
        #wait for init
    
    def close(self):
        self.exit.set()
        self.process.join()
    
    def get(self):
        result = None
        if self.queue.empty():
            result = self.queue.get()
        else:
            while not self.queue.empty():
                result = self.queue.get()
        return result
    
    def _read(self, queue):
        i = 0
        while not self.exit.is_set():
            #read sensor values
            self.ser.reset_input_buffer()
            self.ser.write('a'.encode('utf-8'))
            w, h = 32, 32
            length = 2 * w * h
            input_string = self.ser.read(length)
            x = np.frombuffer(input_string, dtype=np.uint8).astype(np.uint16)
            
            #check initialization
            if len(x) != 2048:
                continue
            
            #reshape values
            x = x[0::2] * 32 + x[1::2]
            x = x.reshape(h, w).transpose(1, 0)

            #append queue
            queue.put(x)
            i += 1

class MultiSensors:
    def __init__(self, ports=['COM15', 'COM7'], normalize=False):
        
        self.ports = ports
        self.normalize = normalize
        self.make_sensors()

        self.queue = Manager().Queue()
        self.exit = mp.Event()
        self.process = mp.Process(target=self._read, args=(self.queue,))
        self.fps_monitor = FramerateMonitor()
        self.fps_queue = Manager().Queue()

    def make_sensors(self):
        sensors = []
        for port in self.ports:
            sensors.append(Sensor(port=port, baudrate=500000, timeout=1.0))
        self.sensors = sensors
    
    def init_sensors(self):
        for sensor in self.sensors:
            sensor.start()

        init_values = []
        for sensor in self.sensors:
            x = sensor.get()
            init_values.append(x.astype(np.float32))

        init_values_num = 30
        for k in range(init_values_num):
            for i in range(len(self.sensors)):
                x = self.sensors[i].get()
                init_values[i] += x
        for i in range(len(self.sensors)):
            init_values[i] /= init_values_num
        self.init_values = init_values

        self.process.start()
    
    def _read(self, queue):
        while not self.exit.is_set():
            images = []
            for sensor in self.sensors:
                x = sensor.get()
                images.append(x)
            #normalize
            if self.normalize:
                for i in range(len(images)):
                    image = images[i].astype(np.float32) - self.init_values[i]

                    minv, maxv = -4, 140
                    image[image < minv] = minv
                    image[image > maxv] = maxv
                    image = (image - minv)/(maxv - minv)
                    #image = (image - np.min(image))/(np.max(image) - np.min(image))
                    images[i] = image

            #concat
            if len(images) == 4:
                images[3] = cv2.rotate(images[3], cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
                images[1] = cv2.rotate(images[1], cv2.cv2.ROTATE_90_CLOCKWISE)
                images[2] = cv2.flip(cv2.rotate(images[2], cv2.cv2.ROTATE_90_CLOCKWISE), 1)

                a = np.concatenate((images[0], images[3]))
                b = np.concatenate((images[1], images[2]))

                a = np.transpose(a, (1, 0))
                b = np.transpose(b, (1, 0))

                total_image = np.concatenate((a, b))
            else:
                total_image = np.concatenate(images)
            
            self.fps_monitor.tick()
            self.fps_queue.put(round(self.fps_monitor.getFps()))
            queue.put(total_image)
    
    def get(self):
        result = None
        if self.queue.empty():
            result = self.queue.get()
        else:
           while not self.queue.empty():
               result = self.queue.get()
        return result
    
    def get_all(self):
        if self.queue.empty():
            results = [self.queue.get()]
        else:
            results = []
            while not self.queue.empty():
                results.append(self.queue.get())
        return results

    def getFps(self):
        result = None
        if self.fps_queue.empty():
            result = self.fps_queue.get()
        else:
            while not self.fps_queue.empty():
                result = self.fps_queue.get()
        return result
    
    def close(self):
        self.exit.set()
        self.process.join()
        for sensor in self.sensors:
            sensor.close()

def main(
        max_frame = 100,
        foldername = 'walking_data/name',
        filename = 'test',
        normalize = False
    ):

    sensor = MultiSensors()
    print("initializing sensors...")
    sensor.init_sensors()

    storage = createStorage('hdf5', foldername, filename, AppContext.create(), {'blockSize': 90})

    while storage.frameCount < max_frame:
        total_image = sensor.get()

        #visualize
        visual_image = copy.deepcopy(total_image)
        visual_image *= 255
        visual_image = visual_image.astype(np.uint8)
        visual_image = cv2.resize(visual_image, (500, 500))
        total_image2 = copy.deepcopy(total_image)
        total_image2 %= 700 
        total_image2 = cv2.resize(total_image2.astype(np.uint8), (500, 500))
        cv2.imshow("Pressure", total_image2)
        if cv2.waitKey(1) & 0xff == 27:
            break

        #fps
        fps = sensor.getFps()
        
        #unix timestep
        ts = dataset_tools.getUnixTimestamp()

        #store data
        storage.addFrame(ts, {'pressure': total_image})

        #verbose
        print(f"FPS : {fps}, Frames : {storage.frameCount}, Storage : {foldername}/{storage.getName()}")
    
    sensor.close()


if __name__ == "__main__":
    print()
    name = input("Enter your name :").strip()
    label = input("Enter label :").strip()
    max_frame = int(input("Enter max frame :").strip())

    main(
        max_frame = max_frame,
        foldername = f'walking_data/{name}',
        filename = label,
        normalize = False
    )