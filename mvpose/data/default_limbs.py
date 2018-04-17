import numpy as np

DEFAULT_JOINT_NAMES = [
    'Nose',
    'Neck',
    'Shoulder right',
    'Elbow right',
    'Hand right',
    'Shoulder left',
    'Elbow left',
    'Hand left',
    'Hip right',
    'Knee right',
    'Foot right',
    'Hip left',
    'Knee left',
    'Foot left',
    'Eye right',
    'Eye left',
    'Ear right',
    'Ear left'
]

# assuming 18 joints
DEFAULT_LIMB_SEQ = np.array(
    [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
     [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
     [0,15], [15,17], [2,16], [5,17]], 'int32')
DEFAULT_LIMB_SEQ.setflags(write=False)  # read-only

DEFAULT_MAP_IDX = np.array(
    [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25],
     [0,  1], [2,  3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29],
     [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27]], 'int32')
DEFAULT_MAP_IDX.setflags(write=False)