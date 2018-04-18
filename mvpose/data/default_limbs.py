import numpy as np

DEFAULT_JOINT_NAMES = [
    'Nose',             # 0
    'Neck',
    'Shoulder right',
    'Elbow right',
    'Hand right',
    'Shoulder left',    # 5
    'Elbow left',
    'Hand left',
    'Hip right',
    'Knee right',
    'Foot right',       # 10
    'Hip left',
    'Knee left',
    'Foot left',
    'Eye right',
    'Eye left',         # 15
    'Ear right',
    'Ear left'
]

# [ ... ( low, high) ... ]
DEFAULT_SENSIBLE_LIMB_LENGTH = np.array([
    (30, 300),  # neck - shoulder right
    (30, 300),  # neck - shoulder left
    (150, 400),  # shoulder right - elbow right
    (150, 600),  # elbow right - hand right
    (150, 400),  # shoulder left - elbow left
    (150, 600),  # elbow left - hand left
    (200, 800),  # neck - hip right
    (200, 800),  # hip right - knee right
    (200, 800),  # knee right - foot right
    (200, 800),  # neck - hip left
    (200, 800),  # hip left - knee left
    (200, 800),  # knee left - foot left,
    (50, 450),  # neck - nose
    (5, 150),  # nose - eye right
    (5, 150),  # eye right - ear right
    (5, 150),  # nose - eye left
    (5, 150),  # eye left - ear left
    (0, 400),  # shoulder right - ear right
    (0, 400)  # shoulder left - ear left
])
DEFAULT_SENSIBLE_LIMB_LENGTH.setflags(write=False)  # read-only

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

assert len(DEFAULT_SENSIBLE_LIMB_LENGTH) == len(DEFAULT_LIMB_SEQ)
assert len(DEFAULT_MAP_IDX) == len(DEFAULT_LIMB_SEQ)