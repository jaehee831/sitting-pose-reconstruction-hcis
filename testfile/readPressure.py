import numpy as np
import h5py

path = './recordings/Touch_COM3.hdf5'
f = h5py.File(path, 'r')
fc = f['frame_count'][0]
touch_ts = np.array(f['ts'][:fc])
data = np.array(f['pressure'][:fc]).astype(np.float32)
