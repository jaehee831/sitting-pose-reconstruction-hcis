import sys; sys.path.insert(0, '..')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from viz.Vizualizer import Vizualizer
from app.AppContext import AppContext
from common import image_tools

PRESSURE_MIN = 0
PRESSURE_MAX = 1000
USE_LOG = True
from IPython import embed
import pickle
# base = pickle.load( open( "C:/Users/alyss/arduino_driver/constant_error_test2.p", "rb" ) )

class VizualizerTouch(Vizualizer):

    # Implementation details
    
    def render(self, sample, auxSamples = None):
        '''
        Renders the content, returns image.
        '''
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        isRight = re.match(r'.*left.*', topic) is None
        pressure = data['pressure']#-base
        pressure = (pressure.astype(np.float32) - PRESSURE_MIN) / (PRESSURE_MAX - PRESSURE_MIN)
        pressure = np.clip(pressure, 0, 1)
        if USE_LOG:
            pressure = np.log(pressure + 1) / np.log(2.0)

        #colormap = matplotlib.cm.get_cmap('jet')
        #im = colormap(pressure)
        #im = (np.clip(im[...,:3], 0, 1) * 255).astype(np.uint8)
        im = cv2.applyColorMap((np.clip(pressure, 0, 1) * 255).astype('uint8'), cv2.COLORMAP_JET)
        
        #import pdb; pdb.set_trace()
        #plt.imshow(im); plt.show()
        #im = im[...,::-1]

        im = image_tools.fitImageToBounds(im, self.resolution, upscale = True, interpolation = cv2.INTER_NEAREST)
        caption = '[%s] %06d (%.3f s)|Range=%03d(%03d)-%03d(%03d)' % (
            topic, frame, ts, data['pressure'].min(), PRESSURE_MIN, data['pressure'].max(), PRESSURE_MAX)
        cv2.putText(im, caption, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), thickness = 1, lineType = cv2.LINE_AA)
        im = image_tools.resizeImageLetterBox(im, self.resolution, interpolation = cv2.INTER_NEAREST)
        return im

    def supports(self, sample, auxSamples = None):
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        if not 'pressure' in data:
            return False
        return True


if __name__ == "__main__":   
    ctx = AppContext.create() 
    ex = VizualizerTouch(ctx = ctx)

    from reader.Reader import Reader
    from reader.ReaderHDF5 import ReaderHDF5
    r = ReaderHDF5(inputFile = r'D:\pkellnho\kitchen\Recordings\open_bottle\touch_right.hdf5', ctx = ctx)
    
    for i, (ts, data) in enumerate(r.read()):
        imViz = ex.render(r.topic, i, ts - r.ts[0], data)
        cv2.imshow('VizualizerTouch', imViz)
        if cv2.waitKey(1) & 0xff == 27:
            break
