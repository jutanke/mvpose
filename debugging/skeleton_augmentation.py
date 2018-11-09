import json
Settings = json.load(open('../settings.txt'))
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from cselect import color as cs
import sys
sys.path.insert(0,'../')
sys.path.insert(0,'../samples')
from mvpose.data import shelf
from time import time

root = Settings['data_root']
root = join(root, 'pak')
tmp = Settings['tmp']

import mvpose.data.kth_football2 as kth
from mvpose import pose
from mvpose.settings import get_settings
from paf_loader import Loader
from mvpose.evaluation import pcp

frame = 300

loader = Loader(with_gpu=False)
import mvpose.data.skeleton_augmentation as ska

Im, Y, Calib = shelf.get(root, frame)

params = get_settings(
    scale_to_mm=1000,
    ms_radius=400,
    ms_between_distance=100,
    max_epi_distance=40,
    gp_max_radius=60,
    pp_conflict_overlap=0.8
)

heatmaps, pafs = loader.load_confidence_map_and_paf(
    'shelf', Im, frame, dir=Settings['tmp'])
detections = pose.estimate(Calib, heatmaps, pafs,
                           settings=params, debug=False)

model_path = '../data/model_poseprediction.h5'

if len(detections) > 0:
    print("L", len(detections))
    for d in detections:
        print("d", len(d))

gen = ska.LimbGenerator(model_path, params.scale_to_mm)
detections2 = gen.apply(detections)

print('------ second try ------')
print("L", len(detections2))
for d in detections2:
    print("d", len(d))
    # for q in d:
    #     print("q", len(q))


# detections3 = gen.apply(detections)
#
# print('------ second try ------')
# print("L", len(detections3))
# for d in detections3:
#     print("d", len(d))
#     # for q in d:
#     #     print("q", len(q))
