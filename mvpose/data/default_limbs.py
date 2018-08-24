import numpy as np

DEFAULT_JOINT_NAMES = [
    'Nose',             # 0
    'Neck',
    'Shoulder right',   # 2
    'Elbow right',      # 3
    'Hand right',       # 4
    'Shoulder left',    # 5
    'Elbow left',
    'Hand left',        # 7
    'Hip right',        # 8
    'Knee right',
    'Foot right',       # 10
    'Hip left',
    'Knee left',        # 12
    'Foot left',        # 13
    'Eye right',
    'Eye left',         # 15
    'Ear right',
    'Ear left'          # 17
]

# GT joints:
# * head (0)                # 0
# * neck (1)
# * left-shoulder (2)
# * left-elbow (3)
# * left-hand (4)
# * right-shoulder (5)      # 5
# * right-elbow (6)
# * right-hand (7)
# * left-hip (8)
# * left-knee (9)
# * left-foot (10)          # 10
# * right-hip (11)
# * right-knee (12)
# * right-foot (13)
DEFAULT_JOINT_TO_GT_JOINT = np.array([
    0,
    1,
    5,   6,  7,  # right arm
    2,   3,  4,  # left arm
    11, 12, 13,  # right leg
    8,   9, 10,  # left leg
    0, 0, 0, 0   # eye - ear
])
DEFAULT_JOINT_TO_GT_JOINT.setflags(write=False)  # read-only


DEFAULT_SYMMETRIC_JOINTS = np.array([
    (2, 5), (3, 6), (4, 7),
    (8, 11), (9, 12), (10, 13),
    (14, 15), (16, 17)
])
DEFAULT_SYMMETRIC_JOINTS.setflags(write=False)  # read-only


# [ ... ( low, high) ... ]
DEFAULT_SENSIBLE_LIMB_LENGTH = np.array([
    (30, 400),  # neck - shoulder right             # 0
    (30, 400),  # neck - shoulder left
    (150, 500),  # shoulder right - elbow right
    (150, 550),  # elbow right - hand right
    (150, 500),  # shoulder left - elbow left
    (150, 550),  # elbow left - hand left           # 5
    (200, 800),  # neck - hip right
    (200, 600),  # hip right - knee right
    (200, 600),  # knee right - foot right
    (200, 800),  # neck - hip left
    (200, 600),  # hip left - knee left             # 10
    (200, 600),  # knee left - foot left,
    (50, 400),  # neck - nose
    (5, 200),  # nose - eye right
    (5, 200),  # eye right - ear right
    (5, 200),  # nose - eye left                    # 15
    (5, 200),  # eye left - ear left
    (0, 550),  # shoulder right - ear right
    (0, 550)  # shoulder left - ear left
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