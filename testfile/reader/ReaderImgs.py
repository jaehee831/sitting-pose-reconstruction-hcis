import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import multiprocessing as mp
import cv2

from reader.Reader import Reader
from app.AppContext import AppContext
from common import dataset_tools

LAZY_READ = True

class ReaderImgs(Reader):
    
    # Implementation details
    
    
    def _init(self):
        self.framesPath = self.inputFile
        self.tsFilename = os.path.join('%s.txt' % self.inputFile)
        if not os.path.isfile(self.tsFilename):
            raise RuntimeError('\tMissing TS file %s!' % self.tsFilename)
        self.log('TS for %s will be read from %s.' % (self.framesPath, self.tsFilename))

        # Read ts
        with open(self.tsFilename,'r') as tsFile:
            self.ts = np.array([float(line.rstrip('\n')) for line in tsFile])
        if len(self.ts) == 0:
            raise RuntimeError('\tCould not read TS file %s!' % self.tsFilename)


    def _release(self):
        pass


    def _read(self):
        
        for i, ts in enumerate(self.ts):
            imPath = os.path.join(self.framesPath, '%03dk' % (i // 1000), '%06d.jpg' % i)
            if not os.path.isfile(imPath):
                raise RuntimeError('Missing frame image "%s".' % imPath)

            if LAZY_READ:
                yield ts, {'im_file': imPath}
                continue

            im = cv2.imread(imPath)
            if im is None:
                raise RuntimeError('Failed to read frame image "%s".' % imPath)

            yield ts, {'im': im}


    @staticmethod
    def supports(inputFile):
        tsFilename = os.path.join('%s.txt' % inputFile)
        if not os.path.isfile(tsFilename):
            return False
        return True




if __name__ == "__main__":    
    ex = ReaderImgs(inputFile = r'C:\pkellnho\PDT\FastTemp\kitchen_rec\pupil_world', ctx = AppContext.create())
    for ts, data in ex.read():
        print('TS = %.6f' % ts)
        cv2.imshow('Image', data['im'])
        if cv2.waitKey(1) & 0xff == 27:
            break


    