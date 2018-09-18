# mvpose
multiple view pose estimation

<img src="https://user-images.githubusercontent.com/831215/45680464-44690d00-bb3b-11e8-87a7-fc5cc2cb6997.png" 
width="500">
<img src="https://user-images.githubusercontent.com/831215/45680466-44690d00-bb3b-11e8-876c-74651b4e5f64.png" 
width="500">
<img src="https://user-images.githubusercontent.com/831215/45680468-4501a380-bb3b-11e8-9f90-e4cf85e8349b.png" 
width="500">
<img src="https://user-images.githubusercontent.com/831215/45680467-4501a380-bb3b-11e8-8ccc-7217780286c4.png"
width="500">


## Install
The software is developed using Python 3.6 and we recommend to use Anaconda Python.
The following additional (non-standard) libraries are needed:

**scikit image**: image manipulation
```bash
conda install -c anaconda scikit-image 
```

**google or-tools**: for solving optimization problems
```bash
pip install --user --upgrade ortools
```

**poseestimation**: for the part affinity fields
```bash
pip install git+https://github.com/jutanke/easy_multi_person_pose_estimation
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
(**Remarks**: The version of OpenCV does not matter too much as long as it is 3.X. However, 
the version bundled with anaconda (I tested with cv2 version 3.1) is apparently not 
compiled against ffmpeg so you cannot load video files. if you want to be on the 'safe' side
compile it on your own!)
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

**person-re-id**: for re-identifying persons
```bash
pip install git+https://github.com/jutanke/person_reid.git
```

### Optional

**cselect**: select colors
```bash
pip install git+https://github.com/jutanke/cselect.git
```
