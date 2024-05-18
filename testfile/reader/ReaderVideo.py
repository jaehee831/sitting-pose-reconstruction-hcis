import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import multiprocessing as mp
import cv2

from reader.Reader import Reader
from app.AppContext import AppContext
from common import dataset_tools


class ReaderVideo(Reader):
    
    # Implementation details
    
    
    def _init(self):
        self.videoFilename = self.inputFile
        self.tsFilename = re.sub(r'(\.[^\.]*)$', '.txt', self.videoFilename)
        if not os.path.isfile(self.tsFilename):
            raise RuntimeError('\tMissing TS file %s!' % self.tsFilename)
        assert self.videoFilename != self.tsFilename
        self.log('TS for %s will be read from %s.' % (self.videoFilename, self.tsFilename))

        # Read ts
        with open(self.tsFilename,'r') as tsFile:
            self.ts = np.array([float(line.rstrip('\n')) for line in tsFile])
        if len(self.ts) == 0:
            raise RuntimeError('\tCould not read TS file %s!' % self.tsFilename)


    def _release(self):
        pass


    def _read(self):
        cap = cv2.VideoCapture(self.videoFilename)
        if not cap:
            raise RuntimeError('\tCannot open %s as a video file!' % self.videoFilename)

        totalFramesEst = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #if fps <= 0:
        #    fps = 30.0
        totalDurationEst = totalFramesEst / fps
        self.log('\tFramerate in video file = %.3f FPS' % fps) # not important, use timestamps instead

        frameIndex = 0
        prevPrintTime = time.time()
        while cap.isOpened():
            ret, im = cap.read()
            if not ret:
                #self.log('Error reading frame.') # Done
                break

            if frameIndex >= len(self.ts):
                raise RuntimeError('Error reading timestamp.')

            ts = self.ts[frameIndex]

            frameIndex += 1

            yield ts, {'im': im}

        cap.release()
    
    
    @staticmethod
    def supports(inputFile):
        if not re.match('.*\.((avi)|(mp4))$', inputFile):
            return False            
        tsFilename = re.sub(r'(\.[^\.]*)$', '.txt', inputFile)
        if not os.path.isfile(tsFilename):
            return False
        return True



if __name__ == "__main__":    
    ex = ReaderVideo(inputFile = r'D:\pkellnho\kitchen\Recordings\test\Webcam_0.avi', ctx = AppContext.create())
    for ts, data in ex.read():
        print('TS = %.6f' % ts)
        cv2.imshow('Image', data['im'])
        if cv2.waitKey(1) & 0xff == 27:
            break


    