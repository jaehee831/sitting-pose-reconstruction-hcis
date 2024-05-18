import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, json, traceback
import numpy as np
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from app.AppContext import AppContext
from app.AppOptions import AppOptions
from common import myglobals, input_tools, dataset_tools

from sensors.Sensor import createSensor


def getParams():
    parser = argparse.ArgumentParser(description='Recorder.')
    parser.add_argument('--config', help='Config JSON file.', default='configs/default.json')
    parser.add_argument('--outputPath', help='Where to store data', default='./recordings/' + str(datetime.datetime.now().strftime(r"rec_%Y-%m-%d_%H-%M-%S")))
    parser.add_argument('--viz', type=input_tools.str2bool, nargs='?', const=True, default=False, help="Debug mode.")
    args = parser.parse_args()

    return args


class Recorder(object):

    def __init__(self):
        self.init()
        super(Recorder, self).__init__()

    def init(self):
        self.ctx = AppContext.create()
        self.log('Initializing...')

        # Config
        self.opts = getParams()
        self.config = self.loadConfig(self.opts.config)
        self.printOptions()

        self.running = False

        self.initSensors()
        
        self.log('Initialized.')

    def release(self):
        self.log('Releasing...')
        
        for sensor in self.sensors:
            sensor.stop()

        cv2.destroyAllWindows()


    def run(self):
        self.log('Starting recording...')

        for sensor in self.sensors:
            sensor.run()

        dummyImg = np.zeros((256,256,3), np.uint8)

        self.running = True
        self.startTS = dataset_tools.getUnixTimestamp()
        t0 = time.time()
        while self.running:
            if time.time() - t0 >= 1.0:
                self.printStatus()
                t0 = time.time()

            # Check status of other sensors
            for sensor in self.sensors:
                if not sensor.running.value:
                    self.log('Sensor %s has terminated. Terminating recording...' % sensor.getName())
                    self.running = False

            if not self.opts.viz:
                # GUI placeholder to capture keys
                cv2.imshow('Recorder', dummyImg)
                if cv2.waitKey(1) & 0xff == 27:
                    self.log('Detected user termination command.')
                    self.running = False

        self.release()

        self.log('Terminated.')

    def initSensors(self):
        self.log('Initializing sensors...')

        self.sensors = []
        for sensorConfig in self.config['sensors']:
            sensorType = sensorConfig['type']
            if sensorType[0] == '#':
                continue # commented out => skip

            params = sensorConfig['params'] # dictionary
            params['outputPath'] = self.opts.outputPath
            params['viz'] = self.opts.viz

            sensor = createSensor(sensorType, self.ctx, params)
            
            self.sensors += [sensor]



        # self.sensors += [Microphone(self.ctx, AppOptions(Microphone.getParser()).getDefault({'outputPath': self.opts.outputPath, 'gain': 4.0, 'viz': self.opts.viz}))]
        
        # self.sensors += [Webcam(self.ctx, AppOptions(Webcam.getParser()).getDefault({'outputPath': self.opts.outputPath, 'cam': 0, 'w': 1280, 'h': 720, 'fps': 10.0, 'storage': 'video', 'viz': self.opts.viz}))]
        # #self.sensors += [Webcam(self.ctx, AppOptions(Webcam.getParser()).getDefault({'outputPath': self.opts.outputPath, 'cam': 1, 'w': 1280, 'h': 720, 'viz': self.opts.viz}))]
        
        # self.sensors += [TouchSensor(self.ctx, AppOptions(TouchSensor.getParser()).getDefault({'outputPath': self.opts.outputPath, 'port': 'COM16', 'viz': self.opts.viz}))]
        # self.sensors += [TouchSensor(self.ctx, AppOptions(TouchSensor.getParser()).getDefault({'outputPath': self.opts.outputPath, 'port': 'COM17', 'viz': self.opts.viz}))]
        
        # self.sensors += [FlexPointGlove(self.ctx, AppOptions(FlexPointGlove.getParser()).getDefault({'outputPath': self.opts.outputPath, 'port': 'COM4', 'hand': 'left', 'viz': self.opts.viz}))]
        # self.sensors += [FlexPointGlove(self.ctx, AppOptions(FlexPointGlove.getParser()).getDefault({'outputPath': self.opts.outputPath, 'port': 'COM18', 'hand': 'right', 'viz': self.opts.viz}))]

    def printStatus(self):
        dt = dataset_tools.getUnixTimestamp() - self.startTS
        msg = '%.3f s || ' % dt

        for sensor in self.sensors:
            msg += '%s (%d @ %.2fHz) | ' % (sensor.getName(), sensor.getFrameCount(), sensor.getFPS())
        self.log(msg)



    def printOptions(self):
        self.log('Main Parameters:')
        for k,v in vars(self.opts).items():
            self.log('\t>> ' + k + ': ' + str(v))
        self.log('--------------------------------')


    def log(self, message):
        self.ctx.log('[Recorder] %s' % message)

    def loadConfig(self, configFile):
        self.log('Loading config from %s...' % configFile)
        try:
            with open(configFile, 'r') as f:
                config = json.load(f)
        except Exception as e:
            self.log('Error while loading config.')
            self.log(traceback.format_exc())
            raise e

        return config




    @staticmethod
    def make():        
        random.seed(65323 + time.time() + os.getpid())
        np.random.seed(int(23125 + time.time() + os.getpid()))

        ex = Recorder()
        ex.run()

        
if __name__ == "__main__":    
    Recorder.make()
    