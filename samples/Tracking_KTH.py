import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
Settings = json.load(open('../settings.txt'))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os.path import join
from cselect import color as cs
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../samples')
sys.path.insert(0,'../debugging')
from mvpose.data import shelf
from time import time

data_root = Settings['data_root']
tmp = Settings['tmp']

import mvpose.data.kth_football2 as kth
from mvpose import pose
from mvpose.settings import get_settings
from paf_loader import Loader
from mvpose.evaluation import pcp
from mvpose.plot.limbs import draw_mscoco_human3d
from mvpose.baseline.baseline import estimate
from openpose import OpenPose
from mvpose.baseline.tracking import Track

seq1_zipname = 'player2sequence1.zip'
seq1_dir = 'Sequence 1'

pe = OpenPose(tmp=tmp)


from mvpose.baseline.tracking import tracking
from time import time

Calib = []
poses_per_frame = []

_start = time()
#for frame in range(0, 214):
end_frame = 60
for frame in range(0, end_frame):
    Im, calib, pos2d, pos3d = kth.get(
        data_root, seq1_zipname, seq1_dir, frame, player=2)
    Calib.append(calib)
    name = 'cvpr_kth_' + seq1_zipname
    predictions = pe.predict(Im, name, frame)
    poses_per_frame.append(predictions)
_end = time()
print('elapsed', _end - _start)

tracks = tracking(Calib, poses_per_frame)

track = tracks[0]

# track.interpolate()
track_ = Track.smoothing(track, sigma=1.7)
#
#
# print(track.frames)
#
# print("q")
# print(track_.frames)
#
# exit(1)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

for frame in range(0, end_frame):
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.plot([0, 0], [-2, 2], [0, 0], color='black', alpha=0.5)
    ax.plot([-2, 2], [0, 0], [0, 0], color='black', alpha=0.5)
    ax.set_title('frame ' + str(frame))

    person = track.get_by_frame(frame)
    if person is not None:
        draw_mscoco_human3d(ax, person, 'red')

    person = track_.get_by_frame(frame)
    if person is not None:
        draw_mscoco_human3d(ax, person, 'blue')

    plt.pause(1/15)
