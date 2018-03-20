""" transform public datasets
*
* head (1)
* neck (2)
* left-shoulder (3)
* left-elbow (4)
* left-hand (5)
* right-shoulder (6)
* right-elbow (7)
* right-hand (8)
* left-hip (9)
* left-knee (10)
* left-foot (11)
* right-hip (12)
* right-knee (13)
* right-foot (14)
"""
import cv2
import numpy as np
import json
from os import listdir
from scipy.ndimage import imread
from os.path import join, isdir, isfile
from pak.datasets.UMPM import UMPM
import h5py

# ========= OpenPose ========


def transform_from_openpose(Y):
    """ transforms the data from the openpose system into
        our reduced set:


    :param Y: (n, m, 18, 2)
    :return:
    """
    #   1    2     3     4     5     6      7    8     9     10    11    12     13    14   15    16     17    18    19
    # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne, Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
    result = []
    for frame in Y:
        new_frame = []
        for user in frame:
            new_frame.append((
                user[0,0:2],
                user[1,0:2],
                user[5,0:2],
                user[6,0:2],
                user[7,0:2],
                user[2,0:2],
                user[3,0:2],
                user[4,0:2],
                user[11,0:2],
                user[12,0:2],
                user[13,0:2],
                user[8,0:2],
                user[9,0:2],
                user[10,0:2]
            ))
        result.append(np.array(new_frame))
    return result


# ========= CMU =========

def get_from_cmu_panoptic(cmu_root, seq_name):
    """ Gets the data from the cmu panoptic dataset.
        Due to how the authors of the dataset want it to
        be handled we do not download it on our own..
        instead it must be downloaded by hand and put in
        the folder given by the cmu_root
    :param cmu_root:
    :param seq_name:
    :return:
    """
    seq_dir = join(cmu_root, seq_name); assert isdir(seq_dir)
    X_fmmap = join(seq_dir, 'X.hdf5')
    is_X_memmapped = isfile(X_fmmap)

    vga_skel_json_path = join(seq_dir, 'vgaPose3d_stage1')
    vga_img_path = join(seq_dir, 'vgaImgs')
    assert isdir(vga_skel_json_path) and isdir(vga_img_path)

    with open(join(seq_dir, 'calibration_{0}.json'.format(seq_name))) as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k, cam in cameras.items():
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3, 1))

    Calib = []
    # get all the videos
    all_videos = sorted([d for d in listdir(vga_img_path) if isdir(join(vga_img_path,d))])
    nbr_videos = len(all_videos)
    w = 640; h = 480  # vga resolution!
    # get shortest video
    nbr_frames = min([len(listdir(join(vga_img_path, v))) for v in all_videos])
    shape = (nbr_videos, nbr_frames, h, w, 3)

    if not is_X_memmapped:  # then we generate it!
        #X = np.memmap(X_fmmap, dtype='uint8', mode='w+', shape=shape)
        f = h5py.File(X_fmmap, 'w')
        X = f.create_dataset('X', shape, dtype='uint8')

    for i, video in enumerate(all_videos):
        # -- calibration --
        cam_key = tuple([int(v) for v in video.split('_')])
        assert cam_key in cameras, 'Cannot find tuple ' + str(cam_key) + ' in cameras'
        cam = {
            'K': cameras[cam_key]['K'],
            'tvec': cameras[cam_key]['t'],
            'rvec': cv2.Rodrigues(cameras[cam_key]['R'])[0],
            'distCoeff': cameras[cam_key]['distCoef']
        }
        Calib.append(cam)

        print('handling video', video)

        # -- load all frames --
        if not is_X_memmapped:  # load the data
            video_loc = join(vga_img_path, video)
            all_videos = listdir(video_loc); assert len(all_videos) >= nbr_frames
            for j, frame in enumerate(sorted(all_videos)[0:nbr_frames]):
                I = imread(join(video_loc, frame))
                X[i,j] = I

    if not is_X_memmapped:  # flush it to file
        f.flush()
        f.close()
        #del X

    f = h5py.File(X_fmmap, 'r')
    X = f['X']

    #X = np.memmap(X_fmmap, dtype='uint8', mode='r', shape=shape)

    return X, Calib


# ========= UMPM =========

def get_from_umpm(root, video_name, user, pwd):
    """ Gets the data of the given UMPM video transformed
        to the mvpose standard
    :param root: data root for all pak datasets
    :param video_name: name of the UMPM video (e.g. "p2_free_1")
    :param user: username for UMPM website
    :param pwd: password for UMPM website
    :return: X, Y, Calib
    """
    umpm = UMPM(root, user, pwd)
    X_, Y, Calib_ = umpm.get_data(video_name)
    X = []
    Calib = []
    for cam in ['l', 'r', 's', 'f']:
        X.append(X_[cam])
        Calib.append(Calib_[cam])

    return X, transform_umpm(Y), Calib


def transform_umpm(Y):
    """
    takes as input the Y from umpm.get_data('...') and converts it into the
    mvpose standard. In this case we just need to drop the 'torso' element
    :param Y: {np.array}, [frames, 15*#persons,5] (x,y,z,_,_)
    :return:
    """
    frames, total_joints, _ = Y.shape
    assert total_joints % 15 == 0, 'wrong total number of joints:' + str(total_joints)

    nbr_persons = int(total_joints / 15)

    # [ frames , joints*#persons , (x,y,z,pid) ]
    Y_mvpose = np.zeros((frames, 14 * nbr_persons, 4))

    umpm_joint_order = [2, 1, 6, 7, 8, 3, 4, 5, 10, 12, 14, 9, 11, 13]

    for f in range(frames):
        # as all individuals are always visible in all frames we can
        # do this:
        for pid in range(nbr_persons):
            mvpose_start = pid * 14
            umpm_start = pid * 15
            for mvpose_pos, umpm_pos in enumerate(umpm_joint_order):
                j = mvpose_pos +  mvpose_start
                j_ = umpm_pos + umpm_start

                Y_mvpose[f, j, 0] = Y[f, j_, 0]
                Y_mvpose[f, j, 1] = Y[f, j_, 1]
                Y_mvpose[f, j, 2] = Y[f, j_, 2]
                Y_mvpose[f, j, 3] = pid

    return Y_mvpose
