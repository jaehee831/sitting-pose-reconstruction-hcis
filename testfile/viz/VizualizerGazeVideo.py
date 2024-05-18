import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback
import numpy as np
import multiprocessing as mp
import cv2

from viz.Vizualizer import Vizualizer
from app.AppContext import AppContext
from common import image_tools

class VizualizerGazeVideo(Vizualizer):

    # Implementation details
    
    def render(self, sample, auxSamples = None):
        '''
        Renders the content, returns image.
        '''
        if 'im' in sample['data']:
            im = sample['data']['im']
        elif 'im_file' in sample['data']:
            im = cv2.imread(sample['data']['im_file'])
        else:
            raise RuntimeError('Missign data.')

        im = image_tools.fitImageToBounds(im, self.resolution, upscale = True)

        # Find gaze
        gaze = None
        gazeAge = 0
        for s in auxSamples:
            if 'gaze_2d' in s['data']:
                gaze = s['data']['gaze_2d'].copy()
                gazeAge = abs(s['ts'] - sample['ts'])
                break
        assert not gaze is None

        # Draw gaze
        if gazeAge < 0.5: 
            gaze[1] = 1 - gaze[1]
            gaze = tuple((gaze * [im.shape[1], im.shape[0]]).astype(int))
            cv2.circle(im, gaze, 10, (0, 0, 255), -1, lineType=cv2.LINE_AA)

        im = image_tools.resizeImageLetterBox(im, self.resolution)
        caption = '[%s] %06d (%.3f s)' % (sample['topic'], sample['frame'], sample['ts'])
        cv2.putText(im, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness = 1, lineType = cv2.LINE_AA)


        return im


    def supports(self, sample, auxSamples = None):
        if not 'im' in sample['data'] and not 'im_file' in sample['data']:
            return False
        if auxSamples is None:
            return False

        for s in auxSamples:
            if 'gaze_2d' in s['data']:
                return True
        
        return False




if __name__ == "__main__":   
    ctx = AppContext.create() 
    ex = VizualizerGazeVideo(ctx = ctx)

    from reader.Reader import createReader
    rs = []
    rs += [createReader(inputFile = r'C:\pkellnho\PDT\FastTemp\kitchen_rec\pupil_world', ctx = ctx)]
    rs += [createReader(inputFile = r'C:\pkellnho\PDT\FastTemp\kitchen_rec\pupil_gaze.hdf5', ctx = ctx)]
    rs[1].ts[0] = rs[1].ts[1] - 1e-3

    from reader.Synchronizer import Synchronizer
    s = Synchronizer(rs, ctx)
    
    for i, (ts, samples) in enumerate(s.read()):
        assert ex.supports(samples[0], samples)
        imViz = ex.render(samples[0], samples)
        cv2.imshow('VizualizerGazeVideo', imViz)
        if cv2.waitKey(1) & 0xff == 27:
            break