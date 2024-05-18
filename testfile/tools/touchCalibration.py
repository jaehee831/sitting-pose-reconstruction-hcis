#import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal, threading
import numpy as np
import h5py

import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

#from common import dataset_tools

CLASS_PASSIVE = 0
CLASS_ACTIVE = 1
CLASS_SHORT = 2

def getParser():
        parser = argparse.ArgumentParser(description='TouchCalibration.')
        parser.add_argument('trainFiles', help='Input HDF5 file', nargs='+')
        parser.add_argument('--testFiles', help='Input HDF5 file', default=None)
        return parser


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

class TouchCalibration(object):
    
    # Implementation details
    
    
    def __init__(self):
        super(TouchCalibration, self).__init__()

    def run(self, filenamesTrain, filenamesTest = None):

        previews = []
        
        pressureTrain = self.readPressures(filenamesTrain)

        print('Preview of raw data...')
        previews += [TouchPreview(self, pressureTrain, title = 'Train/before')]

        #import pdb; pdb.set_trace()
        print('Calibrating...')
        gain, bias, mask, classification, thresholds = self.calibrate(pressureTrain)
        pressureTrainCalib = self.transform(pressureTrain, gain, bias, mask)
                
        print('Preview after calibration...')
        previews += [TouchPreview(self, pressureTrainCalib, mask, classification, title = 'Train/after', limits = [0.0, 2.0])]

        # Test
        if not filenamesTest is None:
            pressureTest = self.readPressures(filenamesTest)
            previews += [TouchPreview(self, pressureTest, title = 'Test/before')]
            _, _, maskTest, classificationTest, _ = self.calibrate(pressureTest, thresholds)
            maskTest = np.logical_and(mask, maskTest)
            pressureTestCalib = self.transform(pressureTest, gain, bias, maskTest)
            previews += [TouchPreview(self, pressureTestCalib, maskTest, classificationTest, title = 'Test/after', limits = [0.0, 2.0])]

        for i,p in enumerate(previews):
            p.setPosition([-1920 + 800 * i, 20])

        while True:
            for p in previews:
                p.update()
            #plt.draw()
            #plt.pause(0.001)
            mypause(0.001)


    def readPressures(self, filenames):
        pressures = []
        for i,filename in enumerate(filenames):
            print('Reading %s...' % filename)
            f = h5py.File(filename, 'r')
            keys = list(f.keys())
            pressure = np.array(f['data'], np.float32)
            f.close()
            pressures += [pressure]
        pressures = np.concatenate(pressures, axis=0)
        return pressures


    def calibrate(self, pressure, thresholds = None):

        THR_SHORT = 800
        THR_DEAD = 25

        pMin = np.percentile(pressure, 5, axis=0)
        pMin = np.min(pressure, axis=0)
        pMax = np.max(pressure, axis=0)

        pDiff = pMax - pMin
        gain = 1.0 / pDiff
        bias = -pMin * gain

        maskShort = pMax > THR_SHORT
        maskDead = pDiff < THR_DEAD
        maskBad = np.logical_or(maskShort, maskDead)
        mask = np.logical_not(maskBad)
        #return gain, bias, mask

        if thresholds is None:
            _thresholds = np.zeros((pressure.shape[1], pressure.shape[2], 2), np.float32)
        else:
            _thresholds = thresholds

        mask = np.zeros((pressure.shape[1], pressure.shape[2]), np.bool)
        classification = np.zeros_like(pressure, int)
        for y in range(pressure.shape[1]):
            for x in range(pressure.shape[2]):
                px = pressure[:,y,x]
                if thresholds is None:
                    thr1 = pMin[y,x] + (pMax[y,x] - pMin[y,x]) * 0.2
                    thr2 = pMin[y,x] + (pMax[y,x] - pMin[y,x]) * 0.5
                    _thresholds[y,x,:] = [thr1, thr2]
                else:
                    thr1, thr2 = thresholds[y,x,:]
                isShorted = px > THR_SHORT # Skip shorted frames
                isPassive = np.logical_and(px < thr1, np.logical_not(isShorted))
                isActive = np.logical_and(px >= thr2, np.logical_not(isShorted))
                if np.sum(isActive) < 3:
                    # No active frames
                    classification[:,y,x] = CLASS_PASSIVE
                    mask[y,x] = False
                    continue

                pxMin = np.mean(px[isPassive])
                pxMax = np.mean(px[isActive])
                pxDiff = pxMax - pxMin
                if pxDiff < THR_DEAD:
                    # Signal too weak
                    classification[:,y,x] = CLASS_PASSIVE
                    mask[y,x] = False
                    continue

                classification[:,y,x] = CLASS_SHORT
                classification[isPassive,y,x] = CLASS_PASSIVE
                classification[isActive,y,x] = CLASS_ACTIVE

                gain[y,x] = 1.0 / pxDiff
                bias[y,x] = -pxMin * gain[y,x]
                mask[y,x] = True

        print('Detected %d valid pixels (%.2f%%)' % (np.sum(mask), np.sum(mask) / np.product(mask.shape) * 100))

        return gain, bias, mask, classification, _thresholds
        

    def transform(self, pressure, gain, bias, mask):
        pressureC = pressure * gain + bias
        
        # Apply mask (zero invalid)
        pressureC1 = pressureC.reshape(pressure.shape[0], -1)
        mask1 = mask.reshape(-1)
        pressureC1[:,np.logical_not(mask1)] = 0#np.NaN
        pressureC = np.reshape(pressureC1, pressureC.shape)

        return pressureC



class TouchPreview(object):

    def __init__(self, app, pressure, mask = None, classification = None, title = '', limits = [500, 900]):
        self.app = app
        self.pressure = pressure
        self.mask = mask
        self.classification = classification
        self.title = title
        self.limits = limits
        self.run()
        super(TouchPreview, self).__init__()
        
    def run(self):#, mask = None):
        #if mask is None:
        #    mask = np.ones((self.pressure.shape[1], self.pressure.shape[1]), np.bool)
        
        cmap = 'jet'
        pMin = self.limits[0] #self.pressure.min()
        pMax = self.limits[1] #self.pressure.max()

        self.app.pickCoords = np.array([16,20], int)
        self.shownCoords = np.array([16,20], int)
        trend = self.pressure[:,self.app.pickCoords[1],self.app.pickCoords[0]]

        pLow = np.zeros((self.pressure.shape[1], self.pressure.shape[2]), np.float32)
        pHigh = np.zeros((self.pressure.shape[1], self.pressure.shape[2]), np.float32)
        pMean = np.zeros((self.pressure.shape[1], self.pressure.shape[2]), np.float32)

        if self.classification is None:
            pLow = np.percentile(self.pressure, 5, axis=0)
            pHigh = np.percentile(self.pressure, 99, axis=0)
            pMean = np.mean(self.pressure, axis = 0)
        else:        
            for y in range(self.pressure.shape[1]):
                for x in range(self.pressure.shape[2]):
                    mx = True if self.mask is None else self.mask[y,x]
                    if mx:
                        px = self.pressure[:,y,x]
                        cx = self.classification[:,y,x]

                        cxLow = cx == CLASS_PASSIVE
                        cxHigh = cx == CLASS_ACTIVE
                        cxMean = cx != CLASS_SHORT
                        if np.any(cxLow):
                            pLow[y,x] = np.mean(px[cxLow])
                        if np.any(cxHigh):
                            pHigh[y,x] = np.mean(px[cxHigh])
                        if np.any(cxMean):
                            pMean[y,x] = np.mean(px[cxMean])

        
        self.fig = plt.figure(figsize=(8,8))
        self.fig.suptitle(self.title, fontsize=16)
        self.axMean = self.fig.add_subplot(2,2,1)
        im = self.axMean.imshow(pMean, vmin = pMin, vmax = pMax, cmap = cmap)
        self.axMean.set_title('Pressure mean')

        self.axTrend = self.fig.add_subplot(2,2,2)
        self.plotTrend = self.axTrend.plot(trend)[0]
        plt.ylim((pMin,pMax))
        self.axTrend.set_title('Trend over time')

        self.axMin = self.fig.add_subplot(2,2,3)
        im = self.axMin.imshow(pLow, vmin = pMin, vmax = pMax, cmap = cmap)
        self.axMin.set_title('Pressure low')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.axMin)
        self.fig.colorbar(im, cax = divider.append_axes('right', size='5%', pad=0.05), orientation = 'vertical')

        self.axMax = self.fig.add_subplot(2,2,4)
        self.axMax.imshow(pHigh, vmin = pMin, vmax = pMax, cmap = cmap)
        self.axMax.set_title('Pressure high')

        #cid = self.fig.canvas.mpl_connect('motion_notify_event', self.onMouseMove)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onMouseMove)
        self.update()
        plt.ion()
        plt.show()

    def setPosition(self, position):
        mngr = self.fig.canvas.manager.window
        geom = mngr.geometry()
        mngr.setGeometry(position[0], position[1], geom.width(), geom.height())


    def update(self):
        #print('Showing trend for %s.' % np.array2string(self.pickCoords))
        if not np.all(np.equal(self.shownCoords, self.app.pickCoords)):
            trend = self.pressure[:,self.app.pickCoords[1],self.app.pickCoords[0]]
            self.plotTrend.set_ydata(trend)
            self.shownCoords = self.app.pickCoords
            print('Updated %s => %s' % (self.title, np.array2string(self.shownCoords)))
        self.fig.canvas.draw()

    
    def onMouseMove(self, event):
        if not event.inaxes in [self.axMean, self.axMin, self.axMax]:
            return

        ix, iy = event.xdata, event.ydata
        #print('x = %d, y = %d'%(ix, iy))

        self.app.pickCoords = np.array([np.round(ix), np.round(iy)], int)
        self.update()

if __name__ == "__main__":    
    parser = getParser()
    args = parser.parse_args()
    ex = TouchCalibration()
    ex.run(args.trainFiles, [args.testFiles])


    