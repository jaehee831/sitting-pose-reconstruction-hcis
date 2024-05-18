import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback, copy, itertools
import numpy as np
import multiprocessing as mp
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from viz.VizualizerOpen3D import VizualizerOpen3D
from app.AppContext import AppContext
from common import image_tools

import open3d as o3d

class VizualizerXsens(VizualizerOpen3D):

    # Implementation details

    def _buildMeshes(self, xsensData):

        VIEW_ANGLE_X = 0
        VIEW_ANGLE_Y = 35

        # In meters
        SKELETON_LEN = 0.5
        SKELETON_WIDTH = 0.05
        FLOOR_SIZE = 2.0

        COLOR_SKELETON = [0.6, 0.5, 0.05]
        #COLOR_JOINT = [0.047, 0.792, 0.062]
        COLOR_GROUND = [0.8, 0.8, 0.8]
        COLOR_RIG = [0.156, 0.498, 0.901]

        # Center
        segPositions = xsensData['position']
        massCenter = xsensData['mass_center'][[1,2,0]] # XYZ->ZXY (z-up to y-up)
        massCenter = np.mean(xsensData['position'], axis=0)
        segPositions -= massCenter * [1,0,1] # center XZ plane

        meshes = []

        # Add skeleton
        for i in range(segPositions.shape[0]):
            #mesh = o3d.geometry.create_mesh_cylinder(radius=SKELETON_WIDTH / 2, height=SKELETON_LEN)
            mesh = o3d.geometry.create_mesh_sphere(radius = SKELETON_WIDTH / 2)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(COLOR_SKELETON)
        
            # Make vertical            
            mesh.rotate([np.pi / 2, 0, 0], center = False, type = o3d.geometry.RotationType.XYZ)

            # Rotate
            rotation = xsensData['rotation_euler'][i,:] / 180 * np.pi
            mesh.rotate(rotation, center = False, type = o3d.geometry.RotationType.XYZ)

            # Move to position
            pos = segPositions[i,:] * 0.01 # cm -> m
            pos = pos * [-1,1,1] # RH->LH
            mesh.translate(pos)


            # Change view angle
            #mesh.rotate([VIEW_ANGLE_X / 180.0 * np.pi, VIEW_ANGLE_Y / 180.0 * np.pi, 0], center = False, type = o3d.geometry.RotationType.XYZ)

            meshes += [mesh]

        # Add ground plane
        mesh = o3d.geometry.create_mesh_box(FLOOR_SIZE, 0.0001, FLOOR_SIZE)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(COLOR_GROUND)
        mesh.translate([-0.5*FLOOR_SIZE, 0, -0.5*FLOOR_SIZE])
        meshes += [mesh]

        # Create rig to stabilize everything a bit
        RIG_SIZE = [2, 2, 2]
        for pos in itertools.product(range(2), repeat=3):
            pos = (np.array(pos, np.float32) - [0.5, 0, 0.5]) * RIG_SIZE
            mesh = o3d.geometry.create_mesh_sphere(radius = 0.05)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(COLOR_RIG)
            mesh.translate(pos)
            meshes += [mesh] 
        #sys.exit(0)

        # Origin
        mesh = o3d.geometry.create_mesh_sphere(radius = 0.05)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.1, 0.5])
        meshes += [mesh] 

        return meshes

        


    def supports(self, sample, auxSamples = None):
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        if not 'position' in data:
            return False
        if not 'rotation_euler' in data:
            return False
        return True


if __name__ == "__main__":   
    ctx = AppContext.create() 
    ex = VizualizerXsens(ctx = ctx, opts={'interactive': False})

    from reader.Reader import Reader
    from reader.ReaderHDF5 import ReaderHDF5
    r = ReaderHDF5(inputFile = r'D:\pkellnho\kitchen\Recordings\walking\xsens.hdf5', ctx = ctx)
    r.shuffleMode = True
    
    for i, (ts, data) in enumerate(r.read()):
        print('%d/%d' % (i, len(r)))
        imViz = ex.render(r.topic, i, ts - r.ts[0], data)
        cv2.imshow('VizualizerXsens', imViz)
        if cv2.waitKey(1) & 0xff == 27:
            break