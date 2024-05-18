import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import multiprocessing as mp
import serial

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
import cv2

from sensors.Sensor import Sensor
from app.AppContext import AppContext
from app.AppOptions import AppOptions
from common import dataset_tools, input_tools

from viz.VizualizerTouch import VizualizerTouch
from IPython import embed
import pickle

# callibrate == True
# constant_error = pickle.load( open( "C:/Users/alyss/arduino_driver/constant_error_large_glove.p", "rb" ) )
class TouchSensor(Sensor):

    @staticmethod
    def getParser():
        parser = Sensor.getParser()
        parser.add_argument('--port', help='COM port', default='COM15') # port is a device name: depending on operating system. e.g. /dev/ttyUSB0 on GNU/Linux or COM3 on Windows.
        parser.add_argument('--baudrate', type=int, help='Baudrate', default=500000)
        parser.add_argument('--hand', help='Use left or right', default='right')
        parser.add_argument('--w', type=int, help='Resolution width', default=32)
        parser.add_argument('--h', type=int, help='Resolution height', default=32)
        parser.add_argument('--blockSize', type=int, help='Storage block size', default=2**14)
        parser.set_defaults(storage = 'hdf5')
        return parser

    # Implementation details

    def getName(self):
        return self.opts.name if len(self.opts.name) > 0 else 'Touch_%s' % str(self.opts.port)

    def _initThread(self):
        # Setup input
        self.ser = serial.Serial(self.opts.port, baudrate=self.opts.baudrate, timeout=1.0)
        assert self.ser.is_open, 'Failed to open COM port!'
        #self.ser.write(b'99') # Why???
        #time.sleep(1)

        if self.opts.viz:
            resolution = [600, 600]
            if self.opts.viz_w > 0 and self.opts.viz_h > 0:
                resolution = [self.opts.viz_w, self.opts.viz_h]
            self.vis = VizualizerTouch(self.ctx, resolution=resolution)
            # def press(event):
            #     if event.key == 'escape':
            #         self.running.value = False
            #         self.log('Detected user termination command.')

            # self.fig = plt.figure(figsize=(8,8))
            # #fig, ax = plt.subplots()
            # self.fig.canvas.mpl_connect('key_press_event', press)
            # self.vizIm = plt.imshow(np.ones((self.opts.h, self.opts.w), np.uint16) * 550, cmap='gray')
            # plt.colorbar()
            # plt.clim(550, 650)

    def _releaseThread(self):
        self.ser.close()

    def readPressure(self):
        # Request readout
        self.ser.reset_input_buffer() # Remove the confirmation 'w' sent by the sensor
        self.ser.write('a'.encode('utf-8')) # Request data from the sensor

        # Receive data
        w, h = self.opts.w, self.opts.h
        length = 2 * w * h
        input_string = self.ser.read(length)
        x = np.frombuffer(input_string, dtype=np.uint8).astype(np.uint16)
        if not len(input_string) == length:
            self.log("Only got %d values => Drop frame." % len(input_string))
            return None
            
        x = x[0::2] * 32 + x[1::2]
        x = x.reshape(h, w).transpose(1, 0)
        return x
    
    def remove_noise(self):
        #This function will take in the values of the different pixels for the first 30seconds and subtract the median noise
        noise = []
        for timer in range(100):
            noise.append(self.readPressure())
        noise_mean = np.mean(noise)
        return noise_mean
    
    def modifiedPressure(self):
        return self.readPressure()-noise_mean
        
    # def hci(self):
        # pressure = self.readPressure()
        # if np.mean(pressure[0:5,0:5])>600:
            # return 
        # elif np.mean(pressure[9:14,9:14])>600:
            # return

    def _read(self):
        # time.sleep(0.01)
        pressure = self.readPressure()
        ts = dataset_tools.getUnixTimestamp()  

        if pressure is None:
            return None, 0
        else:
            # noise = []
            # timer = 800
            # var1 = pressure.astype(np.float)
            # divideby = 1
            # for t in range(1,timer):
                # press = self.readPressure()
                # if press is None:
                    # print( "Pressure was none")
                # else:
                    # divideby = divideby+1
                    # press_tmp = press.copy()
                    # # press_tmp[press_tmp > 980] = np.min(press_tmp)
                    
                    # var1 = var1+press_tmp

                    # print(t,np.amax(press_tmp),np.amin(press_tmp),np.mean(press_tmp))
                    # print(np.max(var1/t), np.min(var1/t))
            # constant_error = var1/divideby
            # pickle.dump(constant_error, open("C:/Users/alyss/arduino_driver/constant_error_large_glove.p","wb"))
            pressure = pressure

            # hand = np.zeros((32,32))    
            
            # pressure = pressure-constant_error - 20
            # hand[2:32,:27] = 1
            # hand[0:2,0:5] = 1000
            # hand[0:22,22:] =1000
            # # hand[:,30:] =1000
            # hand[26:32,22:] = 1000
            # # hand[2:4,0:4] = 1000
            # hand[0:13,4:6] = 1000
            # hand[0:13,10] = 1000
            # hand[0:13,15:18] = 1000
            
            # pressure = (pressure+hand)
            
            #kuka mask
            # hand = np.zeros((32,32))
            # hand[:,0:18] = 1000
            # hand[15:,:] = 1000
            # pressure = (pressure+hand)
                
            
            #Apply averaging and magnifying filter here
            
            
            #Compute median here
            
            
            # print(np.amax(pressure),np.amin(pressure))        
            # print("Starts here")
            #print(np.amax(pressure))
            #print(np.amin(pressure))
            # print('ha', np.min(constant_error), np.max(constant_error))
            # print(np.amax(pressure-constant_error))
            # print(np.amin(pressure-constant_error), np.argmin(pressure-constant_error))
            # print(np.amax(constant_error))
            # print(np.amin(constant_error))
            
            
        return {'pressure': pressure}, ts


    def _viz(self, ts, data):  
        im = self.vis.render({'topic': 'touch_%s' % self.opts.hand, 'frame': self.getFrameCount(), 'ts': ts, 'data': data})
        self.showVizImage(im)

        # self.vizIm.set_data(data['pressure'])
        # #plt.draw()
        # plt.pause(1e-7)


    @staticmethod
    def make():        
        random.seed(65323 + time.time() + os.getpid())
        np.random.seed(int(23125 + time.time() + os.getpid()))
        ex = TouchSensor(ctx = AppContext.create(), opts = AppOptions(TouchSensor.getParser()).getCmd())
        # noise_mean_initial = ex.remove_noise()
        # embed()
        ex.run()
        ex.monitor()

        
if __name__ == "__main__":
    TouchSensor.make()
    
