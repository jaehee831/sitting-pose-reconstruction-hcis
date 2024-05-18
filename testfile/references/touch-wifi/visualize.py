import os, sys
import collections
import random
import time
import math
import numpy as np
import queue as Queue
import threading
import struct
import socket
import select

import cv2


from datetime import datetime
from dateutil import tz

# XXX: put in common module
LOG_SENSORD_UDP_IP = "0.0.0.0"
LOG_SENSORD_UDP_PORT = 1234


LOG_SENSORD_UDP_CTRL_PORT = 1235
LOG_SENSORD_ADDRESS = (LOG_SENSORD_UDP_IP, LOG_SENSORD_UDP_PORT)
#LOG_BASE_FILENAME = "/data/logs/sensor.log"
#LOG_CURRENT_FILENAME = LOG_BASE_FILENAME + ".0"
#FIRMWARE_UDP_IP = "0.0.0.0"
#FIRMWARE_UDP_PORT = 11060

class DynamicPlotter():
    WC = 32
    HC = 32
    def __init__(self, sampleinterval = 0.1, timewindow = 10.0, size=(1024, 640)):
        self.pressarray =  np.zeros((32,32), dtype=np.uint32)
        self.recvfrom = None
        #import pdb; pdb.set_trace()
        t = threading.Thread(target=self.producer)  #, args=(self.dataQueue,self.queueLock))
        t.daemon = True
        t.start()
        t2 = threading.Thread(target=self.i2cset)  #, args=(self.dataQueue,self.queueLock))
        t2.daemon = True
        t2.start()      
        #self.i2cset()  
        


    def connect(self):
        self.serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.serverSock.bind((LOG_SENSORD_UDP_IP, LOG_SENSORD_UDP_PORT))

    def i2cset(self) :
        while True:
            strin = input()
            nums = strin.split()
            if not self.recvfrom is None:
                sendSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                datByte = struct.pack('BBBB',  0x11, int(nums[0]), int(nums[1]),0 )
                #print(datByte, self.recvfrom)
                sendSock.sendto(datByte, (self.recvfrom[0], LOG_SENSORD_UDP_CTRL_PORT))



    def producer(self):
        self.connect()
        print("connected")        
        while True:
            #line = self.readLogLine()
            # Read sensor values from firmware
            try:
                import pdb; pdb.set_trace()
                data, addr = self.serverSock.recvfrom(1024)
                self.recvfrom = addr
            except socket.timeout:
                # Give chance to quit from loop if shutdown was requested,
                # otherwise just keep trying to read from socket.
                #self.readingsLabel.setText("Sensor readings: TIMEOUT (firmware down?)")
                continue
            data_channel = struct.unpack('I', data[0:4])[0] & 0xFFF
            self.pressarray[data_channel, :] = np.frombuffer(data, dtype=np.uint32, count=self.WC, offset=4)
            #np.set_printoptions(formatter={'int':hex})
            if data_channel == 0:
                #print(self.pressarray)                
                src = cv2.resize( self.pressarray / 1024, (256,256), interpolation = cv2.INTER_NEAREST)                
                src = np.uint8(src) #.astype('uint8')
                #print(src)
                im_color = cv2.applyColorMap(src, cv2.COLORMAP_JET)
                cv2.imshow('frame', im_color)
                if cv2.waitKey(1) & 0xFF == 'q':
                    break
            


if __name__ == '__main__':
    m = DynamicPlotter(sampleinterval = 0.05, timewindow = 20.0)
    while True:
        time.sleep(1)
