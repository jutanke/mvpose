import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from mvpose.data import epfl_campus
from time import time

root = Settings['data_root']
root = join(root, 'pak')
tmp = Settings['tmp']

import shutil
from os.path import isdir, join
from os import makedirs

import mvpose.data.kth_football2 as kth
from mvpose import pose
from mvpose.settings import get_settings
from paf_loader import Loader
from mvpose.evaluation import pcp

import mvpose.data.kth_football2 as kth
from mvpose import pose
from mvpose.settings import get_settings
from paf_loader import Loader
from mvpose.evaluation import pcp
from mvpose.plot.limbs import draw_mscoco_human, draw_mscoco_human2d

from openpose import OpenPose

pe = OpenPose(tmp=tmp, peak_threshold=0.05)
valid_frames = list(range(350, 470)) + list(range(650, 750))

Calib = []
poses_per_frame = []
Pos3d = {}
Imgs = {}
peak_threshold = 0.05

_start = time()
for frame in valid_frames:
    Im, Y, calib = epfl_campus.get(root, frame)
    Imgs[frame] = Im
    Calib.append(calib)
    Pos3d[frame] = Y
    predictions = pe.predict(Im, 'cvpr_campus' + str(peak_threshold),
                             frame)
    poses_per_frame.append(predictions)
_end = time()
print('elapsed', _end - _start)

from mvpose.baseline.tracking import tracking, Track
_start = time()
tracks = tracking(Calib, poses_per_frame,
                  epi_threshold=20,
                  scale_to_mm=1000,
                  max_distance_between_tracks=200,
                  actual_frames=valid_frames,
                  min_track_length=10,
                  merge_distance=80,
                  last_seen_delay=5)
_end = time()
print('elapsed', _end - _start)

print("#tracks", len(tracks))

for track in tracks:
    print(len(track))

Enable_Smoothing = True

# -- smooth tracks --
if Enable_Smoothing:
    _start = time()
    tracks_ = []
    for track in tracks:
        track = Track.smoothing(track,
                                sigma=1.7,
                                interpolation_range=50)
        tracks_.append(track)
    tracks = tracks_
    _end = time()
    print("elapsed", _end - _start)
# -----------------


# ---- video ----
CREATE_VIDEO = True
if Enable_Smoothing:
    video_dir = './video_campus'
else:
    video_dir = './video_campus_smooth'
if CREATE_VIDEO:
    if isdir(video_dir):
        shutil.rmtree(video_dir)
    makedirs(video_dir)
# ---- video ----

colors = ['red', 'blue', 'green', 'teal', 'yellow', 'gray', 'cornflowerblue', 'cyan', 'black', 'white', 'magenta']

fig = plt.figure(figsize=(16, 12))
Axs = [fig.add_subplot(2, 3, 1 + cid) for cid in range(len(calib))]


for i, t in enumerate(valid_frames):
    print('handle frame ' + str(t + 1))

    Im = Imgs[t]
    for cid, cam in enumerate(calib):
        ax = Axs[cid]
        ax.clear()
        ax.axis('off')
        im = Im[cid]
        h, w, _ = im.shape
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        ax.imshow(im, alpha=0.6)
        ax.set_title("Frame " + str(t))

        for tid, track in enumerate(tracks):
            pose = track.get_by_frame(t)
            if pose is None:
                continue

            draw_mscoco_human(ax, pose, cam,
                              alpha=0.5,
                              color=colors[tid],
                              linewidth=3)

    if CREATE_VIDEO:
        #plt.tight_layout()
        # extent = ax.get_window_extent().transformed(
        #     fig.dpi_scale_trans.inverted())
        fig.savefig(join(video_dir, 'out%05d.png' % (i + 1)))
    else:
        plt.pause(1/10)