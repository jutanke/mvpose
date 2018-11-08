import numpy as np
import cv2
from os import listdir
from os.path import join, isdir
import json
from scipy.ndimage import imread
from mvpose.geometry.camera import ProjectiveCamera


def transform_y(Y):
    """
    :param Y: [ (pid, [(x,y,z), (x,y,z), ...] ]
    :return:
    """
    reorder = [
        (0, 1),  # nose
        (1, 0),  # neck
        (2, 9),  # shoulder right
        (3, 10),
        (4, 11),
        (5, 3),  # shoulder left
        (6, 4),
        (7, 5),
        (8, 12),  # hip right
        (9, 13),
        (10, 14),
        (11, 6),  # hip left
        (12, 7),
        (13, 8),
        (14, 17),  # eye right
        (15, 15),
        (16, 18),  # ear left
        (17, 16)

    ]  # (ours, cmu)

    result = []
    for pid, skel in Y:
        new_skel = np.zeros((18, 4))
        for ours, cmu in reorder:
            new_skel[ours] = skel[cmu]
        result.append((pid, new_skel))
    return result


def get(cmu_root, seq_name, panels, nodes, frame=0):
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

    vga_skel_json_path = join(seq_dir, 'hdPose3d_stage1_coco19')
    vga_img_path = join(seq_dir, 'hdImgs')
    assert isdir(vga_skel_json_path) and isdir(vga_img_path)

    # sometimes the skel are stored in a additional folder "hd"
    vga_skel_json_path_hd = join(vga_skel_json_path, 'hd')
    if isdir(vga_skel_json_path_hd):
        vga_skel_json_path = vga_skel_json_path_hd

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
        assert isdir(join(vga_img_path, v), ), 'does not exist:' + join(vga_img_path, v)

    nbr_videos = len(all_videos)
    #w = 640; h = 480  # vga resolution!
    w = 1920; h = 1080  # hd resolution
    # get shortest videonbr_frames
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
        #
        # cam = {
        #     'K': cam['K'],
        #     'tvec': cam['t'],
        #     'rvec': cv2.Rodrigues(cam['R'])[0],
        #     'distCoeff': cam['distCoef']
        # }
        cam = ProjectiveCamera(
            cam['K'], cv2.Rodrigues(cam['R'])[0], cam['t'],
            cam['distCoef'], w, h)
        Calib.append(cam)

    Y = []
    try:
        skel_json_fname = vga_skel_json_path + '/body3DScene_{0:08d}.json'.format(frame)
        with open(skel_json_fname) as dfile:
            bframe = json.load(dfile)

            for body in bframe['bodies']:
                pid = body['id']
                #skel = np.array(body['joints15']).reshape((-1, 4))
                skel = np.array(body['joints19']).reshape((-1, 4))

                Y.append((pid, skel))
    except IOError as e:
        print('Error reading {0}\n'.format(skel_json_fname) + e.strerror)

    return X, Y, Calib

