import numpy as np
import mvpose.geometry.geometry as gm
from mvpose.geometry.camera import ProjectiveCamera
from pak.datasets.UMPM import UMPM


def get(root, video_name, user, pwd):
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
        _, h, w, _ = X_[cam].shape
        K, rvec, tvec, distCoef = gm.get_camera_parameters(Calib_[cam])
        Calib.append(ProjectiveCamera(K, rvec, tvec, distCoef, w, h))

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