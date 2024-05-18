import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

from viz.Vizualizer import Vizualizer
from app.AppContext import AppContext
from common import image_tools

class VizualizerVideo(Vizualizer):

    # Implementation details
    
    def render(self, sample, auxSamples = None):
        '''
        Renders the content, returns image.
        '''
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        if 'im' in sample['data']:
            im = sample['data']['im']
        elif 'im_file' in sample['data']:
            im = cv2.imread(sample['data']['im_file'])
        else:
            raise RuntimeError('Missign data.')
        im = image_tools.resizeImageLetterBox(im, self.resolution)

        caption = '[%s] %06d (%.3f s)' % (topic, frame, ts)
        cv2.putText(im, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness = 1, lineType = cv2.LINE_AA)
        return im

    def supports(self, sample, auxSamples = None):
        #topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        if not 'im' in sample['data'] and not 'im_file' in sample['data']:
            return False
        return True




if __name__ == "__main__":   
    ctx = AppContext.create() 
    ex = VizualizerVideo(ctx = ctx)

    from reader.Reader import Reader
    #from reader.ReaderVideo import ReaderVideo as ReaderEx
    #r = ReaderVideo(inputFile = r'D:\pkellnho\kitchen\Recordings\test\Webcam_0.avi', ctx = ctx)
    from reader.ReaderImgs import ReaderImgs as ReaderEx
    r = ReaderEx(inputFile = r'C:\pkellnho\PDT\FastTemp\kitchen_rec\pupil_world', ctx = ctx)
    
    for i, (ts, data) in enumerate(r.read()):
        imViz = ex.render(r.topic, i, ts - r.ts[0], data)
        cv2.imshow('VizualizerVideo', imViz)
        if cv2.waitKey(1) & 0xff == 27:
            break