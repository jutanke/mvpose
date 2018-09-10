import numpy as np
import cv2
from os import listdir
from os.path import join, isdir
import json
from scipy.ndimage import imread
from mvpose.geometry.camera import ProjectiveCamera


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
        assert isdir(join(vga_img_path, v), ), 'does not exist:' + join(vga_img_path, v)

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