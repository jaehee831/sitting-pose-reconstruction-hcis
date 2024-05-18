import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

from app.AppContext import AppContext
from common import image_tools

from viz.VizualizerVideo import VizualizerVideo
from viz.VizualizerTouch import VizualizerTouch
from viz.VizualizerFlexPoint import VizualizerFlexPoint
from viz.VizualizerXsens import VizualizerXsens
from viz.VizualizerGazeVideo import VizualizerGazeVideo
VIZ_CLASSES = [VizualizerTouch, VizualizerFlexPoint, VizualizerXsens, VizualizerGazeVideo, VizualizerVideo]

class VizMaster(object):
    '''
    Find correct viz for every topic.
    '''

    def __init__(self, ctx, resolution):
        self.ctx = ctx
        self.resolution = np.array(resolution)
        self.init()
        super(VizMaster, self).__init__()

    def init(self):
        self.vizs = []
        for VIZ_CLASS in VIZ_CLASSES:
            self.vizs += [VIZ_CLASS(self.ctx, resolution=[32,32])]
        
    def release(self):
        for viz in self.vizs:
            viz.release()
        self.vizs = []



    def render(self, samples, order = None):
        '''
        Tiles all visualizations.
        '''
        # Mark valid
        isValid = np.array([not self.getRenderer(sample, samples) is None for sample in samples])

        # Decide layout
        N = np.sum(isValid)
        targetAspect = self.resolution[0] / float(self.resolution[1])
        aspect = 1.0
        gridAspect = targetAspect / aspect
        h = math.ceil(math.exp((math.log(N) - math.log(gridAspect)) / 2))
        w = math.ceil(N / h)
        layout = [w, h]
        tileSize = self.resolution // layout
            

        # Render
        if order is None:
            order = np.arange(len(samples))
        imgs = []
        dummy = np.zeros((tileSize[1], tileSize[0], 3), np.uint8)
        for ii in order:
            i = order[ii]
            if not isValid[i]:
                continue
            if i < 0:
                imgs += [dummy]
                continue
            sample = samples[i]
            im = self.renderTopic(tileSize, sample, samples)
            if im is None:
                self.log('No renderer for topic %s!' % sample['topic'])
                imgs += [dummy]
                continue
            imgs += [im]

        #     cv2.imshow(sample['topic'], im)
        # cv2.waitKey(0)
            
        # Merge
        im = image_tools.gridImages(imgs, layout = layout)

        # Check size (may not match in case of rounding)
        im = image_tools.letterBoxImage(im, self.resolution)

        return im

    def getRenderer(self, sample, auxSamples):
        '''
        Tries to find a suitable renderer.
        '''
        for viz in self.vizs:
            if viz.supports(sample, auxSamples):
                return viz
        return None
            

    def renderTopic(self, resolution, sample, auxSamples):
        viz = self.getRenderer(sample, auxSamples)
        viz.setResolution(resolution)
        return viz.render(sample, auxSamples)


    def log(self, msg):
        self.ctx.log('[%s] %s' % (self.__class__.__name__, msg))