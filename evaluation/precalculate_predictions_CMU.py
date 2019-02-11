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

root = Settings['CMU']['data_root']
tmp = Settings['tmp']

import mvpose.data.kth_football2 as kth
from mvpose.evaluation import pcp
from mvpose.plot.limbs import draw_mscoco_human, draw_mscoco_human2d
from mvpose.data import cmu_panoptic


from openpose import OpenPose

print("TMP", tmp)
pe = OpenPose(tmp=tmp)


Frames_Pizza = list(range(1000, 8000))
Frames_Ultimatum = list(range(300, 6880))
Frames_Haggling = list(range(4209, 5315)) + list(range(6440, 8200)) + list(range(26510, 27168))

sequences = ['160906_pizza1', '160422_ultimatum1', '160224_haggling1']

print(len(Frames_Haggling))


def apply(seq_name, frame):
    print('handle ' + seq_name, frame)
    _start = time()
    nodes = [0, 1, 2, 3, 4]
    panels = [0, 0, 0, 0, 0]
    Im, Y, Calib = cmu_panoptic.get(root, seq_name,
                                    panels, nodes, frame=frame)
    pe.predict(Im, 'cvpr_cmu' + seq_name, frame)
    _end = time()
    print("\telapsed", _end - _start)


while True:
    if len(Frames_Pizza) > 0:

        frame = Frames_Pizza.pop(0)
        seq_name = '160906_pizza1'
        apply(seq_name, frame)
    if len(Frames_Ultimatum) > 0:
        frame = Frames_Ultimatum.pop(0)
        seq_name = '160422_ultimatum1'
        apply(seq_name, frame)
    if len(Frames_Haggling) > 0:
        frame = Frames_Haggling.pop(0)
        seq_name = '160224_haggling1'
        apply(seq_name, frame)

    if len(Frames_Pizza) == 0 and len(Frames_Ultimatum) == 0 and len(Frames_Haggling) == 0:
        break
