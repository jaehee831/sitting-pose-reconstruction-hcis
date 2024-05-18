import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback, copy, itertools
import numpy as np
import multiprocessing as mp
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from viz.Vizualizer import Vizualizer
from app.AppContext import AppContext
from common import image_tools

import open3d as o3d

class VizualizerOpen3D(Vizualizer):

    # Implementation details

    def _init(self):
        #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        if not 'interactive' in self.opts:
            self.opts['interactive'] = False

        self.vis = None
        
        self.meshes = []
        self.camera = None

    def buildScene(self, data):

        self.disposeScene()

        self.meshes = self._buildMeshes(data)
        for mesh in self.meshes:
            self.vis.add_geometry(mesh)  

        # Store camera
        vc = self.vis.get_view_control()
        if self.camera is None:
            self.camera = vc.convert_to_pinhole_camera_parameters()

        # Restore camera
        vc.convert_from_pinhole_camera_parameters(self.camera)

        

    def disposeScene(self):
        for mesh in self.meshes:
            self.vis.remove_geometry(mesh)
        self.meshes = []
    

    def _release(self):
        self.vis.destroy_window()
        self.vis = None

    
    def render(self, sample, auxSamples = None):
        '''
        Renders the content, returns image.
        '''
        # Lazy init
        if self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='%s_Open3D' % self.getName(), width=self.resolution[0], height=self.resolution[1], visible=self.opts['interactive'])
            ro = self.vis.get_render_option()
            #ro.show_coordinate_frame = True
        
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        data = self._preprocessData(topic, frame, ts, data)
        self.buildScene(data)

        # Render
        self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

        if self.opts['interactive']:
            # Return dummy
            return np.zeros((64, 64, 3), np.uint8)
            
        # Retrieve rendered image
        im = self.vis.capture_screen_float_buffer(do_render=False)
        im = np.asarray(im)[...,::-1]
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        
        self.renderCaption(im, topic, frame, ts, data)
        return im

    def renderCaption(self, im, topic, frame, ts, data):
        caption = '[%s] %06d (%.3f s)' % (topic, frame, ts)
        cv2.putText(im, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness = 1, lineType = cv2.LINE_AA)



    # To implement
    def _preprocessData(self, topic, frame, ts, data):
        return data

    def _buildMeshes(self, data):
        raise RuntimeError()

