import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

from app.AppContext import AppContext


class Vizualizer(object):

    def __init__(self, ctx, resolution = (600, 600), opts = {}):
        self.ctx = ctx
        self.resolution = resolution
        self.opts = opts
        self.init()
        super(Vizualizer, self).__init__()

    def init(self):
        self._init()

    def release(self):
        self._release()

    def getName(self):
        return '%s' % (self.__class__.__name__)

    def log(self, msg):
        self.ctx.log('[%s] %s' % (self.getName(), msg))


    # Implementation details

    def _init(self):
        pass

    def _release(self):
        pass
    
    def render(self, sample, auxSamples = None):
        '''
        Renders the content, returns image.
        '''
        return []

    def supports(self, sample, auxSamples = None):
        return False

    def setResolution(self, resolution):
        self.resolution = resolution