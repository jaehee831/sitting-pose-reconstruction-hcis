# Touch sensor recordings

## Credits

The code was written by [Petr Kellnhofer](https://people.csail.mit.edu/pkellnho/) using some pieces of code from [Yunzhu Li](https://people.csail.mit.edu/liyunzhu/). The code was written as a part of human modeling project under supervision of [Wojciech Matusik](https://people.csail.mit.edu/wojciech/) and [Antonio Torralba](https://web.mit.edu/torralba/www/) at MIT CSAIL, 2019.

### Quick start

To record with one pcb board: 
```
python sensors/TouchSensor.py
```

To record with multiple pcb boards:
```
python main.py --config configs/default.json --outputPath [path to output directory]
```
where all sensors are defined in ''default.json''

To replay:
```
python player.py [inputPath] --sinkPath [where to store the vizualization] --viz
```
(`inputPath` is the recording `--outputPath`).

After you realize it is not so easy, read rest of this document.

### Features

Currently the system records pressure frame from the [touch sensor](https://github.com/Erkil1452/touch) through USB-COM interface

### Output formats

We currently support these output formats for recording of the sensor outputs to HDD:
- `HDF5` database for other data through [h5py](https://www.h5py.org/)


## Installation

In order to use this SW package several steps need to be taken. We need to install all packages needed for the Python environment but also install all 3rd party SW packages and install the external HW and configure it acccording to the vendors' manuals.

### System
The SW has been tested on Windows 10 x64 Pro. Some sensors will run under Linux and this code is multi-platform but not all 3rd party SW is supported.

### Python environment
The system has been implemented and tested to run with Python 3.6 (x64) under Windows 10 x64 Pro. These packages are required to be installed (see [requirements.txt](requirements.txt)):
- h5py  (>= 2.9.0) - for HDF5 input/output
- matplotlib (>= 2.2.2) - for plotting
- msgpack (>= 0.6.1) - for ZMQ
- numpy (>= 1.16.4) - for vector data types and computation
- open3d (>= 0.7.0.0) - for 3D rendering
- opencv (>= 4.1.0) - for video input/output and GUI
- PyAudio (>= 0.2.11) - for audio input
- pyzmq (>= 18.0.1) - for PupilLabs communication
- scipy (>= 1.2.1) - for Matlab I/O




### Touch sensor

We use our own [touch sensor](https://github.com/Erkil1452/touch) to measure force feedback of the hand. It communicates with the computer using USB-serial port similarly to the `Flexpoint` sensor. Again, proper COM port has to be setup in the configuration of this application. We currently use the original [Subra's board](http://humangrasp.io) but with a modified firmware made by [Yunzhu Li](https://people.csail.mit.edu/liyunzhu/). The Adruino Nano code is in [arduino_programs/tactile_32x32/tactile_32x32.ino](arduino_programs/tactile_32x32/tactile_32x32.ino).




### Other requirements

Some of the non-essential features have additional requirements.

`ffmpeg` has to be available in the system path (e.g., callable from command line) in order to enable support for recording playback video encoding.



## Recording

### Third-party SW setup

Make sure to first connect all the HW and run all the 3rd party SW + calibrations as described above (`PupilLabs Capture` and `Xsense MVN`).

### Configuration

The list of all sensors and their configuration is specified by `JSON` configuration files (e.g. [config/default.json](configs/default.json)). The structure of the config file is:
```
sensors
    type
    params
        name
        ...
```
where
- `type` is the sensor class as defined in `sensor\Sensor.py` -> `createSensor()`
- `params` is an array of sensor specific parameters

**Note:** You can comment out a sensor by prefixing its `type` by `#`.

The most common parameters are:
- `name` - names the output file
- `cam`/`port` - determines the input index / serial port / network port
- `storage` - specifies the storage format as defined in `storage/Storage.py` -> `createStorage()`.
- `viz_*` - specifies the position and size of the preview window
- `hand` - specifies which hand the sensor measures (`left` or `right`)
- `w`/`h`/`fps` - specifies the spatial/termporal resolution

You can see complete lists in the static `getParser()` method of each of the individual sensors in `sensors/*`.


### Running the code

To start recording go to the code root directory and call:
```
python main.py --config [path to config JSON] --outputPath [path to output directory] [--viz]
```
The default configuration used is the `config/default.json`.
You can enable "real-time" visualization of the sensors using the switch `--viz` but **DO NOT USE IT FOR REAL RECORDING** as it **significantly** reduces the framerate. Use it only to validate that everything works as expected.




