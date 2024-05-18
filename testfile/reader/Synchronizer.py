import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import multiprocessing as mp
import cv2

from app.AppContext import AppContext
from common import dataset_tools


class Synchronizer(object):

    def __init__(self, readers, ctx, fps = 30.0, keepFull = False):
        self.readers = readers
        self.ctx = ctx
        self.fps = fps
        self.keepFull = keepFull
        self.init()
        super(Synchronizer, self).__init__()


    def init(self):
        assert len(self.readers) > 0

        minT = self.readers[0].ts[0]
        maxT = self.readers[0].ts[-1]
        maxFPS = 0
        for i, reader in enumerate(self.readers):
            if self.keepFull:
                # Take union of ranges
                minT = min(minT, reader.ts[0])
                maxT = max(maxT, reader.ts[-1])
            else:
                # Take intersection of ranges
                minT = max(minT, reader.ts[0])
                maxT = min(maxT, reader.ts[-1])
            maxFPS = max(maxFPS, reader.fps)

        
        self.fps = min(self.fps, maxFPS)
        self.duration = maxT - minT
        nFrames = math.ceil(self.fps * self.duration)
        self.ts = np.linspace(minT, maxT, nFrames)
        self.log('Duration = %.3f s @ %.2f FPS => %d frames' % (self.duration, self.fps, nFrames))

        self.nearestIndices = np.zeros((len(self.ts), len(self.readers)))

        for i, reader in enumerate(self.readers):
            self.nearestIndices[:,i] = dataset_tools.findNearestFrame(self.ts, reader.ts)

    def read(self):
        '''
        Generator function that synchonizes all readers.
        '''
        ptrs = np.zeros((len(self.readers),), int) - 1
        iters = [reader.read() for reader in self.readers]
        samples = [None for reader in self.readers]
        topics = [reader.topic for reader in self.readers]

        for i, ts in enumerate(self.ts):
            #self.log('[%06d/%06d] %.3f seconds (%.2f%%)' % (i, len(self), ts - self.ts[0], i / len(self) * 100.0))

            for j, iter in enumerate(iters):
                while ptrs[j] < self.nearestIndices[i, j]:
                    sample = next(iter)
                    ptrs[j] += 1
                    samples[j] = {
                        'topic': topics[j],
                        'frame': ptrs[j],
                        'ts': sample[0] - self.ts[0],
                        'ts_orig': sample[0],
                        'data': sample[1],
                    }

            yield (ts, samples)
            
    def __len__(self):
        return len(self.ts)    

    def log(self, msg):
        self.ctx.log('[Synchronizer] %s' % (msg))

  