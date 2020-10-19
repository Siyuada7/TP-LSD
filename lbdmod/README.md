# lbdmod

Modified from https://github.com/alexandervakhitov/lbdmod

A line detection & description pipeline based on the EDLines detector and LBD descriptor in the OpenCV implementation.

To use, please 

1. Install dependencies (gcc, OpenCV, for the python wrapper also Python and boost)

2. Customize the CMakeLists.txt, part with Python paths (provide paths to the python and numpy libraries and headers)

3. Use (under Linux)
```
mkdir build
cd build
cmake . ..
make
``` 
Then the library will be built in the root folder, and the wrapper will be installed to the Python *dist-packages* folder.

You can check that it works by running
```
./lbd_mod_test
``` 
from the same folder. After the execution, the file 'test.png' with line detection visualization should appear in this folder.

You can check the python interface by running
```
python ../python/lbdtest.py
```

In the python interface, we represent the set of lines as a OpenCV Mat instance. Each row has 17 entries corresponding to 
line angle, number (class_id), octave, middlepoint's x and y coordinates, response, size, start point's x and y, end point's x and y, start point's x and y in octave, end point's x and y in octave, line length, number of pixels covered by the line (see https://github.com/alexandervakhitov/lbdmod/blob/master/cpp/src/dataconv.cpp). Similarly, line matches are stored.
