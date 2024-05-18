import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import multiprocessing as mp
import cv2

from app.AppContext import AppContext


class Reader(object):

    def __init__(self, inputFile, ctx, opts = None):
        self.inputFile = inputFile
        self.ctx = ctx
        self.opts = opts
        self.init()
        super(Reader, self).__init__()


    def init(self):
        self.ts = []
        self._init()

    def release(self):
        self._release()

    def read(self):
        return self._read()

    def log(self, msg):
        self.ctx.log('[%s] %s' % (self.getName(), msg))

    def getName(self):
        return '%s (%s)' % (self.__class__.__name__, os.path.basename(self.inputFile))

    def __len__(self):
        return len(self.ts)

    @property
    def duration(self):
        return self.ts[-1] - self.ts[0]

    @property
    def fps(self):
        return (len(self) - 1) / self.duration

    @property
    def topic(self):
        matches = re.match(r'(.+)(\.[^\.]*)?$', os.path.basename(self.inputFile))
        if matches is None:
            return 'UNKNOWN'
        return matches.group(1)

    # Implementation details    
    def _init(self):
        pass

    def _release(self):
        pass

    def _read(self):
        yield [], None

    @staticmethod
    def supports(inputFile):
        return True



from reader.ReaderVideo import ReaderVideo
from reader.ReaderImgs import ReaderImgs
from reader.ReaderHDF5 import ReaderHDF5
READERS = [ReaderVideo, ReaderImgs, ReaderHDF5]
READERS = [ReaderHDF5]
    
def createReader(inputFile, ctx, opts = None):
    allOk = True
    for ReaderClass in READERS:
        #import pdb; pdb.set_trace()
        if not ReaderClass.supports(inputFile):
            continue
        try:
            print(inputFile)
            reader = ReaderClass(inputFile, ctx, opts)
        except:     
            print('\t[%s] Unexpected error: %s' % (ReaderClass.__name__, sys.exc_info()))   
            allOk = False    
            continue
        return reader, allOk
    return None, allOk

