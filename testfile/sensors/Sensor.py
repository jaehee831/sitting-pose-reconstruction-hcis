import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

from app.AppContext import AppContext
from app.AppOptions import AppOptions
from app.FramerateMonitor import FramerateMonitor
from storage.Storage import createStorage
from common import dataset_tools, input_tools, image_tools



class Sensor(object):

    @staticmethod
    def getParser():
        parser = argparse.ArgumentParser(description='Sensor.')
        parser.add_argument('--name', help='Camera name', default='')
        parser.add_argument('--outputPath', help='Where to store data', default='./recordings/')
        parser.add_argument('--storage', help='Storage type', default='dummy')
        parser.add_argument('--viz', type=input_tools.str2bool, nargs='?', const=True, default=False, help="Debug mode.")
        parser.add_argument('--viz_x', type=int, default=-2**20, help="Viz window x.")
        parser.add_argument('--viz_y', type=int, default=-2**20, help="Viz window y.")
        parser.add_argument('--viz_w', type=int, default=-2**20, help="Viz window width.")
        parser.add_argument('--viz_h', type=int, default=-2**20, help="Viz window height.")
        parser.add_argument('--sp', type=input_tools.str2bool, nargs='?', const=True, default=False, help="Single process mode.")
        return parser

    def __init__(self, ctx, opts):
        self.ctx = ctx
        self.opts = opts
        self.init()
        super(Sensor, self).__init__()

    def init(self):
        self.log('Initializing...')
        self._printOptions()

        self.running = mp.Value('b', True) 
        self.startTS = mp.Value('d', 0) 
        self.frameCount = mp.Value('i', 0) 
        self.fps = mp.Value('f', -1.0) 

        self.log('Initialized.')


    def run(self):
        self.log('Starting recording...')
        
        self.processes = []
        self.running.value = True
        if self.opts.sp:
            self._run()
        else:
            self.processes.append(mp.Process(target=self._run, args=()))
            self.startTS.value = dataset_tools.getUnixTimestamp()
            for p in self.processes:
                p.start()
        
        self.log('Recording.')


    def stop(self):
        self.log('Stopping recording...')

        self.running.value = False
        for p in self.processes:
            #p.terminate() # do not do! Would not release properly.
            p.join()

        self.log('Stopped.')

    def log(self, msg):
        self.ctx.log('[%s] %s' % (self.getName(), msg))


    def _run(self):
        self.log('Thread: Starting...')
        self.fpsMonitor = FramerateMonitor()
        self.storage = None

        if self.opts.sp:
            # Unsafe for debug
            self._initThread()
            self._initStorage()
        else:
            try:
                self._initThread()
            except Exception as e:
                self.running.value = False
                self.log('Error during intialization of the sensor.')
                self.log(traceback.format_exc())
                return

            try:
                self._initStorage()
            except Exception as e:
                self.running.value = False
                self.log('Error during intialization of the storage.')
                self.log(traceback.format_exc())
                return

        
        if self.opts.viz and self._supportsViz():
            cv2.namedWindow(self.getName(), cv2.WINDOW_AUTOSIZE)
            if self.opts.viz_x > -2**20 and self.opts.viz_y > -2**20:
                cv2.moveWindow(self.getName(), self.opts.viz_x, self.opts.viz_y)

        isFirst = True
        lastViz = time.time()
        lastData = None
        lastValidData = None
        lastTS = 0
        lastValidTS = 0
        while self.running.value:

            if self.opts.viz and self._supportsViz() and not lastValidData is None:
                dViz = time.time() - lastViz 
                if dViz > 0.1 or lastData is None:  
                    self._viz(lastValidTS, lastValidData)
                    lastViz = time.time()

            #self.ctx.log('runRead: Reading frame')
            lastData, lastTS = self._read()   
            if lastData is None:
                time.sleep(0.001)
                continue

            if isFirst:
                self.startTS.value = dataset_tools.getUnixTimestamp()
                isFirst = False

            # Store new data
            self._addFrame(lastTS, lastData)
            lastValidData = lastData
            lastValidTS = lastTS

            # Update FPS
            self.fpsMonitor.tick()
            self.fps.value = self.fpsMonitor.getFps()


            #self.ctx.tick(self.getName())
        if self.opts.viz and self._supportsViz():
            cv2.destroyAllWindows()

        if not self.storage is None:
            self.storage.release()
            
        self._releaseThread()
        self.log('Thread: Terminated.')


    def _initStorage(self):
        self.storage = createStorage(self.opts.storage, self.opts.outputPath, self.getName(), self.ctx, self.opts)
        self.log('Using storage "%s".' % self.storage.getName())

    def _addFrame(self, ts, data):
        '''
        Store one frame.
        '''
        self.storage.addFrame(ts, data)        
        self.frameCount.value = self.storage.frameCount
        
    def _printOptions(self):
        self.log('Parameters:')
        for k,v in vars(self.opts).items():
            self.log('\t>> ' + k + ': ' + str(v))

    def getRunningTime(self):
        return dataset_tools.getUnixTimestamp() - self.startTS.value

    def getFrameCount(self):
        return self.frameCount.value

    def getFPS(self):
        return self.fps.value

    def getFPSTotal(self):
        return self.getFrameCount() / self.getRunningTime()  

    def monitor(self):
        original_sigint = signal.getsignal(signal.SIGINT)
        def signal_handler(sig, frame):
            signal.signal(signal.SIGINT, original_sigint)
            print('You pressed Ctrl+C!')
            ex.stop()
        signal.signal(signal.SIGINT, signal_handler)

        t0 = time.time()
        while self.running.value:
            if time.time() - t0 >= 1.0:
                self.log('T = %.3f s | %d frames | %.2f Hz' % (
                    self.getRunningTime(), self.getFrameCount(), self.getFPS()
                ))
                t0 = time.time()


    def showVizImage(self, im):
        if self.opts.viz_w > 0 and self.opts.viz_h > 0:
            im = image_tools.resizeImageLetterBox(im, [self.opts.viz_w, self.opts.viz_h])
        cv2.imshow(self.getName(), im)

        if cv2.waitKey(1) & 0xff == 27:
            self.log('Detected user termination command.')
            self.running.value = False


    # Implementation details

    def getName(self):
        '''
        Get sensor instance name.
        '''
        return None
   

    def _initThread(self):
        '''
        Acquire resources (sensor).
        '''
        pass

    def _releaseThread(self):
        '''
        Release the sensor.
        '''
        pass

    def _read(self):
        '''
        Read a frame from the sensor.
        '''
        pass


    def _viz(self, ts, data):        
        '''
        Show single frame.
        '''
        pass

    def _supportsViz(self):
        return True


def createSensor(sensorType, ctx, opts):
    SensorClass = None
    if sensorType == 'webcam':
        from sensors.Webcam import Webcam as SensorClass
    elif sensorType == 'mic':
        from sensors.Microphone import Microphone as SensorClass
    elif sensorType == 'touch':
        from sensors.TouchSensor import TouchSensor as SensorClass
    elif sensorType == 'touch2':
        from sensor.TouchSensor2 import TouchSensor2 as SensorClass
    elif sensorType == 'flexpoint':
        from sensors.FlexPointGlove import FlexPointGlove as SensorClass
    elif sensorType == 'pupil_gaze':
        from sensors.PupilLabsGaze import PupilLabsGaze as SensorClass
    elif sensorType == 'pupil_video':
        from sensors.PupilLabsVideo import PupilLabsVideo as SensorClass
    elif sensorType == 'xsens':
        from sensors.XSensSensor import XSensSensor as SensorClass
    elif sensorType == 'scale':
        from sensors.ScaleSensor import ScaleSensor as SensorClass
    else:
        raise RuntimeError('No such Sensor Type "%s"!' % sensorType)

    print(SensorClass)
    sensorParser = SensorClass.getParser()
    print(sensorParser)
    sensorOpts = AppOptions(sensorParser).getDefault(opts)
    print(sensorOpts)
    return SensorClass(ctx, sensorOpts)


