# mvpose
multiple view pose estimation

## Install
The software is developed using Python 3.6 and we recommend to use Anaconda Python.
The following additional (non-standard) libraries are needed:

**google or-tools**: for solving optimization problems
```bash
pip install --user --upgrade ortools
```

**numba**: enables high performance functions in Python using NumPy
```bash
conda install -c numba numba
```

**c3d**: for reading 3d data
```bash
pip install c3d
```

**pak**: for loading some of the datasets
```bash
pip install git+https://github.com/justayak/pppr.git
pip install git+https://github.com/justayak/pak.git
```

**opencv3.X**: for common computer vision tasks (reproject 3d points, etc)
We recommend to compile your own version of OpenCV for your python setup.
At the cmake-stage you can choose whatever additional flags you like (cuda, opencvcontrib, etc.) -- 
however, keep the ones written below as they are. You might need to install
additional packages (e.g. ffmpeg, libpng, etc.) before you are able to successfully
build OpenCV (check out the OpenCV documentation for your OS for further instructions).
```bash
git clone https://github.com/opencv/opencv.git
cd opencv && git checkout 3.4.0
mkdir build && cd build
cmake -DBUILD_opencv_java=OFF \ 
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$(python3 -c "import sys; print(sys.prefix)") \
    -DPYTHON3_EXECUTABLE=$(which python3) \
    -DPYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") .. 
make -j4
make install
```
