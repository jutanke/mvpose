import json
Settings = json.load(open('../settings.txt'))
import matplotlib.pyplot as plt
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
from mvpose.plot.limbs import draw_mscoco_human, draw_mscoco_human2d
from mvpose.baseline.baseline import estimate
from openpose import OpenPose

seq1_zipname = 'player2sequence1.zip'
seq1_dir = 'Sequence 1'

peak_threshold = 0.08
pe = OpenPose(tmp=tmp, peak_threshold=peak_threshold)

from mvpose.baseline.tracking import tracking
from time import time

Calib = []
poses_per_frame = []
Pos3d = []
Ims = []

_start = time()
# for frame in range(0, 214):
end_frame = 100
for frame in range(0, end_frame):
    Im, calib, pos2d, pos3d = kth.get(
        data_root, seq1_zipname, seq1_dir, frame, player=2)
    Calib.append(calib)
    Pos3d.append(pos3d)
    Ims.append(Im)

    txt_add = str(peak_threshold)
    if 0.099 < peak_threshold < 0.101:
        txt_add = ''

    name = 'cvpr_kth_' + seq1_zipname + txt_add
    predictions = pe.predict(Im, name, frame)
    poses_per_frame.append(predictions)
_end = time()
print('elapsed', _end - _start)

tracks = tracking(Calib, poses_per_frame,
                  epi_threshold=110,
                  scale_to_mm=1000,
                  max_distance_between_tracks=200,
                  distance_threshold=200,
                  correct_limb_size=False,
                  merge_distance=200)

track = tracks[0]
from mvpose.baseline.tracking import Track
track = Track.smoothing(track, sigma=2.3, interpolation_range=5)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

for frame in range(end_frame):
    ax.clear()
    ax.imshow(Ims[frame][0])
    pose = track.get_by_frame(frame)
    draw_mscoco_human(ax, pose, cam=Calib[frame][0], color='red')

    plt.pause(1/30)
