import sys; sys.path.insert(0, '.')
import sys, os, re, time, shutil, math, random, datetime, argparse, signal
import numpy as np
import h5py

from reader.Reader import Reader
from app.AppContext import AppContext
from common import dataset_tools



class ReaderHDF5(Reader):
    
    # Implementation details
    
    
    def _init(self):
        self.f = h5py.File(self.inputFile, 'r')
        self.frameCount = self.f['frame_count'][0] # may be lower than the number of records due to pre-allocation
        self.ts = self.f['ts'][:self.frameCount]
        self.keys = list(self.f.keys())
        self.log('Reading keys: %s' % ', '.join(self.keys))
        self.shuffleMode = False
        # import scipy.io as sio
        # data = {}
        # for k in self.keys:
        #     data[k] = np.array(self.f[k])
        # sio.savemat('xsens.mat', data)
        # sys.exit(0)

    def _release(self):
        self.f.close()

    def _read(self):
        inds = np.arange(self.frameCount)
        if self.shuffleMode:
            np.random.shuffle(inds)
        for ii in range(self.frameCount):
            i = inds[ii]
            ts = self.ts[i]
            frame = {}
            for k in self.keys:
                if k == 'frame_count':
                    continue
                frame[k] = self.f[k][i,...]
            yield ts, frame

    @staticmethod
    def supports(inputFile):
        return bool(re.match('.*\.((h5)|(hdf5))$', inputFile))
    

if __name__ == "__main__": 
    ReaderHDF5.supports(r'D:\pkellnho\kitchen\Recordings\test2\xsens.hdf5')
    ex = ReaderHDF5(inputFile = r'D:\pkellnho\kitchen\Recordings\test2\xsens.hdf5', ctx = AppContext.create())
    print('Total %d records.' % len(ex))
    for ts, frame  in ex.read():
        #print('TS = %.6f: %s' % (ts, frame))
        pass
    print('DONE')


    