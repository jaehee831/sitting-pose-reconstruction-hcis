import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, traceback, copy
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

class VizualizerFlexPoint(VizualizerOpen3D):

    # Implementation details

    def _buildMeshes(self, data):

        fpData = data['data']
        isRight = data['isRight']
        hs = 1.0 if isRight else -1.0

        FINGER_SEG_LEN = 1.0
        FINGER_WIDTH = 0.2
        FINGER_SEPARATION_X = 1.8
        FINGER_SEPARATION_Y = 1.1
        PALM_WIDTH = FINGER_WIDTH * FINGER_SEPARATION_X * 3 + FINGER_WIDTH
        VIEW_ANGLE_X = 0
        VIEW_ANGLE_Y = hs * 35

        COLOR_PALM = [0.901, 0.666, 0.156]
        COLOR_PROXIMAL = [0.156, 0.498, 0.901]
        COLOR_DISTAL = [0.047, 0.792, 0.062]

        MAX_JOINT_ANGLE = 90 / 180.0 * np.pi
        jointAngles = (fpData[:10].astype(np.float32) / 2**15) * MAX_JOINT_ANGLE
        #downDir = 
        magField = fpData[13:16].astype(np.float32)
        magRadius = np.linalg.norm(magField)
          
        magTheta = 0*math.atan(magField[1] / (magField[0]+1))
        magPhi = 0*math.acos(magField[2] / magRadius)
       
        meshes = []
        
        for i in range(5):
            # 5 fingers: thumb -> little

            # proximal -> distal
            for j in range(2):
                #mesh = o3d.geometry.create_mesh_coordinate_frame()
                mesh = o3d.geometry.create_mesh_cylinder(radius=FINGER_WIDTH / 2, height=FINGER_SEG_LEN)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(COLOR_PROXIMAL if j == 0 else COLOR_DISTAL)

                # Make vertical
                mesh.rotate([np.pi / 2, 0, 0], center = False, type = o3d.geometry.RotationType.XYZ)
                # Move origin to the base
                mesh.translate([0, FINGER_SEG_LEN / 2, 0])

                # Apply joints from end to bottom
                for k in range(j, -1, -1):
                    angle = 25 / 180 * np.pi
                    angle = jointAngles[(4-i)*2+k]

                    # Rotate in joint
                    mesh.rotate([-angle, 0, 0], center = False, type = o3d.geometry.RotationType.XYZ)
                    
                    # Move up (stack vertically)               
                    if k > 0:
                        posRelY = (FINGER_SEG_LEN * FINGER_SEPARATION_Y)                                
                        mesh.translate([0, posRelY, 0])

                if i == (0 if isRight else 4):
                    # Rotate thumb
                    mesh.rotate([0, 0, hs * np.pi / 4], center = False, type = o3d.geometry.RotationType.XYZ)
                    # Move down "a bit"
                    mesh.translate([0, - 0.5 * FINGER_SEG_LEN, 0])
                    # Move right a bit
                    mesh.translate([hs * 0.5 * FINGER_WIDTH, 0, 0])

                # Move horizontally
                posRelX = i * (FINGER_WIDTH * FINGER_SEPARATION_X)
                mesh.translate([posRelX, 0, 0])

                # Hand orientation
                mesh.rotate([magPhi, magTheta, 0], center = False, type = o3d.geometry.RotationType.XYZ)

                # Change view angle
                mesh.rotate([VIEW_ANGLE_X / 180.0 * np.pi, VIEW_ANGLE_Y / 180.0 * np.pi, 0], center = False, type = o3d.geometry.RotationType.XYZ)


                meshes += [mesh]

        # Add palm
        mesh = o3d.geometry.create_mesh_box(PALM_WIDTH, PALM_WIDTH, FINGER_WIDTH)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(COLOR_PALM)
        mesh.translate([-FINGER_WIDTH / 2, -PALM_WIDTH - (FINGER_SEG_LEN * (FINGER_SEPARATION_Y-1)), 0])
        if isRight:
            mesh.translate([(FINGER_WIDTH * FINGER_SEPARATION_X), 0, 0])

        # Hand orientation
        mesh.rotate([magPhi, magTheta, 0], center = False, type = o3d.geometry.RotationType.XYZ)

        # Change view angle
        mesh.rotate([VIEW_ANGLE_X / 180.0 * np.pi, VIEW_ANGLE_Y / 180.0 * np.pi, 0], center = False, type = o3d.geometry.RotationType.XYZ)

        meshes += [mesh]

        return meshes

    def _preprocessData(self, topic, frame, ts, data):
        isRight = re.match(r'.*left.*', topic) is None
        #self.log('Generating hand %s' % ('right' if isRight else 'left'))
        data['isRight'] = isRight
        return data

    def renderCaption(self, im, topic, frame, ts, data):
        caption = '[%s] (%s) %06d (%.3f s)' % (
            topic, 'right' if data['isRight'] else 'left', frame, ts)
        cv2.putText(im, caption, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness = 1, lineType = cv2.LINE_AA)

    def supports(self, sample, auxSamples = None):
        topic, frame, ts, data = sample['topic'], sample['frame'], sample['ts'], sample['data']
        if not 'data' in data:
            return False
        if not len(data['data']) >= 16:
            return False
        return True


if __name__ == "__main__":   
    ctx = AppContext.create() 
    ex = VizualizerFlexPoint(ctx = ctx)

    from reader.Reader import Reader
    from reader.ReaderHDF5 import ReaderHDF5
    r = ReaderHDF5(inputFile = r'D:\pkellnho\kitchen\Recordings\open_bottle\fp_right.hdf5', ctx = ctx)
    r.shuffleMode = True

    for i, (ts, data) in enumerate(r.read()):
        print('%d/%d' % (i, len(r)))
        imViz = ex.render(r.topic, i, ts - r.ts[0], data)
        cv2.imshow('VizualizerFlexPoint', imViz)
        if cv2.waitKey(1) & 0xff == 27:
            break