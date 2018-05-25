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
from os.path import join, isdir
from pak.datasets.UMPM import UMPM
from mvpose.data.default_limbs import DEFAULT_JOINT_TO_GT_JOINT

# ========= OpenPose ========


def transform_3D_detections(person, joint_to_gt_joint=DEFAULT_JOINT_TO_GT_JOINT):
    """

    :param person: [ (x,y,z,...), ... ] every limb of the person
    :param joint_to_gt_joint: maps the detection point to the ground truth point
    :return: detected person using the same structure as our ground truth
    """
    assert len(person) == len(joint_to_gt_joint)
    transformed_human = [None] * (np.max(joint_to_gt_joint) + 1)

    for joint, gt_loc in zip(person, joint_to_gt_joint):
        if not joint is None:
            ojoint = transformed_human[gt_loc]
            if ojoint is not None:
                transformed_human[gt_loc] = (ojoint + joint) / 2
            else:
                transformed_human[gt_loc] = joint
    return transformed_human


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

def get_from_cmu_panoptic(cmu_root, seq_name, panels, nodes, frame=0):
    """ Gets the data from the cmu panoptic dataset.
        Due to how the authors of the dataset want it to
        be handled we do not download it on our own..
        instead it must be downloaded by hand and put in
        the folder given by the cmu_root
    :param cmu_root:
    :param seq_name:
    :return:
    """
    assert len(panels) > 0 and len(panels) == len(nodes)
    seq_dir = join(cmu_root, seq_name); assert isdir(seq_dir)
    #X_fmmap = join(seq_dir, 'X.hdf5')
    #is_X_memmapped = isfile(X_fmmap)

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
        cam['t'] = np.squeeze(np.array(cam['t']).reshape((3, 1)))

    Calib = []
    # get all the videos
    # all_videos = sorted([d for d in listdir(vga_img_path) if isdir(join(vga_img_path,d))])
    # if len(all_videos) > len(panels):
    #
    all_videos = []
    for p, n in zip(panels, nodes):
        all_videos.append('{0:02d}_{1:02d}'.format(p, n))
    for v in all_videos:
        assert isdir(join(vga_img_path, v))

    nbr_videos = len(all_videos)
    w = 640; h = 480  # vga resolution!
    # get shortest video
    nbr_frames = min([len(listdir(join(vga_img_path, v))) for v in all_videos])
    assert frame < nbr_frames

    X = np.zeros((nbr_videos, h, w, 3), 'uint8')

    cams = zip(panels, nodes)
    sel_cameras = [cameras[cam].copy() for cam in cams]

    for icam, cam in enumerate(sel_cameras):
        image_path = vga_img_path + '/{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'],
                                                                                                                frame)
        im = imread(image_path)
        X[icam] = im

        cam = {
            'K': cam['K'],
            'tvec': cam['t'],
            'rvec': cv2.Rodrigues(cam['R'])[0],
            'distCoeff': cam['distCoef']
        }
        Calib.append(cam)

    Y = []
    try:
        skel_json_fname = vga_skel_json_path + '/body3DScene_{0:08d}.json'.format(frame)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

            for body in bframe['bodies']:
                pid = body['id']
                skel = np.array(body['joints15']).reshape((-1, 4))

                Y.append((pid, skel))
    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname) + e.strerror)

    return X, Y, Calib


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
