from mvpose.baseline.tracking import tracking
from time import time
import json
Settings = json.load(open('../settings.txt'))
import matplotlib.pyplot as plt
import numpy as np
from cselect import color as cs
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../samples')
sys.path.insert(0,'../debugging')
from mvpose.data import shelf
from time import time
import shutil
from os.path import isdir, join
from os import makedirs
root = join(Settings['data_root'], 'pak')
tmp = Settings['tmp']


import mvpose.data.kth_football2 as kth
from mvpose.plot.limbs import draw_mscoco_human, draw_mscoco_human2d

from openpose import OpenPose

pe = OpenPose(tmp=tmp)

poses_per_frame = []
start_frame = 300
end_frame = 450
actual_frames = list(range(start_frame, end_frame))

calib = None
_start = time()
Imgs = {}
for t in range(start_frame, end_frame):
    Im, _, calib = shelf.get(root, t)
    Imgs[t] = Im
    predictions = pe.predict(Im, 'cvpr_shelf', t)
    poses_per_frame.append(predictions)
print('extract poses from ' +\
      str(end_frame - start_frame) + ' frames')
print('\telapsed:', time() - _start)

_start = time()
tracks = tracking(calib, poses_per_frame,
                  actual_frames=actual_frames,
                  epi_threshold=80)
print('extract ' + str(len(tracks)) + ' tracks')
print('\telapsed:', time() - _start)


colors = ['red', 'blue', 'green', 'teal', 'yellow', 'gray']

fig = plt.figure(figsize=(16, 12))
Axs = [fig.add_subplot(2, 3, 1 + cid) for cid in range(len(calib))]


# ---- video ----
CREATE_VIDEO = True
video_dir = './video'
if CREATE_VIDEO:
    if isdir(video_dir):
        shutil.rmtree(video_dir)
    makedirs(video_dir)
# ---- video ----

for i, t in enumerate(actual_frames):
    print('handle frame ' + str(t + 1) + "/" + str(end_frame))

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
