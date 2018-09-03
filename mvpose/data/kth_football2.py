import numpy as np
import numpy.linalg as la
import cv2
from os import makedirs
from os.path import join, isdir, isfile
from pak.util.download import download
from pak.util.unzip import unzip
from math import ceil, floor
from mvpose.geometry.camera import AffineCamera
from mvpose.algorithm.candidate_selection import project_human_to_2d


def get(data_root, seq_zipname, seq_dir, frame=0, player=1):
    """

    :param data_root:
    :param seq_zipname:
    :param seq_dir:
    :param frame: starting at frame 0
    :param player which player
    :return:
    """
    root = join(data_root, 'football2')
    root = join(root, 'player' + str(player))
    if not isdir(root):
        makedirs(root)

    seq_url = 'http://www.csc.kth.se/cvap/cvg/MultiViewFootballData/' + seq_zipname
    seq_dir = join(root, seq_dir)

    if not isdir(seq_dir):
        seq_zip = join(root, seq_zipname)
        if not isfile(seq_zip):
            print('downloading... ', seq_url)
            download(seq_url, seq_zip)

        print('unzipping... ', seq_zip)
        unzip(seq_zip, root)

    pos2d_file = join(seq_dir, 'positions2d.txt')
    pos2d = np.loadtxt(pos2d_file)
    N = 14  # number joints
    C = 3  # number cameras
    T = len(pos2d) / 2 / N / C
    assert floor(T) == ceil(T)
    T = int(T)

    pos2d_result = np.zeros((2, N, C, T))
    counter = 0
    for t in range(T):
        for c in range(C):
            for n in range(N):
                for i in range(2):
                    pos2d_result[i, n, c, t] = pos2d[counter]
                    counter += 1
    pos2d = pos2d_result

    # ~~~ pos3d ~~~
    pos3d_file = join(seq_dir, 'positions3d.txt')
    assert isfile(pos3d_file)
    pos3d = np.loadtxt(pos3d_file)
    pos3d_result = np.zeros((3, N, T))
    assert T == int(len(pos3d) / 3 / N)
    counter = 0
    for t in range(T):
        for n in range(N):
            for i in range(3):
                pos3d_result[i, n, t] = pos3d[counter]
                counter += 1
    pos3d = pos3d_result

    # ~~~ Cameras ~~~
    cam_file = join(seq_dir, 'cameras.txt')
    assert isfile(cam_file)
    cams = np.loadtxt(cam_file)
    cameras = np.zeros((2, 4, C, T))
    assert T == int(len(cams) / 2 / 4 / C)

    counter = 0
    for t in range(T):
        for c in range(C):
            for j in range(4):
                for i in range(2):
                    cameras[i, j, c, t] = cams[counter]
                    counter += 1

    Im = []
    h = -1; w = -1
    for cam in ['Camera 1', 'Camera 2', 'Camera 3']:
        im_dir = join(seq_dir, cam)
        assert isdir(im_dir)
        im_name = join(im_dir, "%05d.png" % (frame+1))
        assert isfile(im_name)
        im = cv2.cvtColor(cv2.imread(im_name), cv2.COLOR_BGR2RGB)
        Im.append(im)

        if w == -1 or h == -1:
            assert h == -1 and w == -1
            h, w, _ = im.shape
        else:
            h_, w_, _ = im.shape
            assert h_ == h and w_ == w

    Im = np.array(Im)

    Calib = []
    for cid in [0, 1, 2]:
        cam = np.zeros((3, 4))
        cam[0:2, :] = cameras[:, :, cid, frame]
        cam[2,3] = 1
        Calib.append(AffineCamera(cam, w, h))

    # h x w x cam
    Pts2d = []
    for cid in [0, 1, 2]:
        d2d = pos2d[:,:,cid, frame]
        Pts2d.append(d2d)

    d3d = pos3d[:, :, frame]

    return Im, Calib, \
           np.transpose(np.array(Pts2d)), np.transpose(d3d)


def draw_limbs2d(ax, person3d, cam, color, print_length=False):
    """
        draws the person onto the screen
    :param ax:
    :param person3d:
    :param cam:
    :param color:
    :param print_length: {boolean} if true: print limb length
    :return:
    """
    assert len(person3d) == 14
    person2d = project_human_to_2d(person3d, cam)
    for p in person2d:
        if p is not None:
            ax.scatter(p[0], p[1], color=color)
    limbs = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
        (9, 3), (2, 8), (8, 12), (9, 12), (12, 13)
    ]
    limb_names = [
        'lower leg (right)',
        'upper leg (right)',
        'pelvis',
        'upper leg (left)',
        'lower leg (left)',
        'lower arm (right)',
        'upper arm (right)',
        'shoulders',
        'upper arm (left)',
        'lower arm (left)',
        'side (left)',
        'side (right)',
        'shoulder-to-chin (right)',
        'shoulder-to-chin (left)',
        'head'
    ]
    assert len(limbs) == len(limb_names)
    for lid, (a, b) in enumerate(limbs):
        if person2d[a] is not None and person2d[b] is not None:
            Ax, Ay = person2d[a]
            Bx, By = person2d[b]
            ax.plot([Ax, Bx], [Ay, By], color=color)
            if print_length:
                print(limb_names[lid] + '= ',
                      la.norm(person3d[a] - person3d[b]))


def transform3d_from_umpm(humans):
    """
        transforms from umpm dataset ot kth
    :param humans:
    :return:
    """
    human_t = []
    for human in humans:
        new_human = [None] * 14
        new_human[13] = human[0]
        new_human[12] = human[1]
        new_human[9] = human[2]
        new_human[10] = human[3]
        new_human[11] = human[4]
        new_human[8] = human[5]
        new_human[7] = human[6]
        new_human[6] = human[7]
        new_human[3] = human[8]
        new_human[4] = human[9]
        new_human[5] = human[10]
        new_human[2] = human[11]
        new_human[1] = human[12]
        new_human[0] = human[13]
        human_t.append(new_human)
    return human_t


def transform3d_from_mscoco(humans):
    """
        transforms the humans in the list from the mscoco
        data structure to the kth football2 structure
    :param humans: [ [ (x,y,z), ... ] * n_limbs ] * n_humans
    :return:
    """
    # R_ANKLE       0
    # R_KNEE        1
    # R_HIP         2
    # L_HIP         3
    # L_KNEE        4
    # L_ANKLE       5
    # R_WRIST       6
    # R_ELBOW       7
    # R_SHOULDER    8
    # L_SHOULDER    9
    # L_ELBOW       10
    # L_WRIST       11
    # BOTTOM_HEAD   12
    # TOP_HEAD      13
    human_t = []

    for human in humans:
        new_human = [None] * 14
        new_human[0] = human[10]
        new_human[1] = human[9]
        new_human[2] = human[8]
        new_human[3] = human[11]
        new_human[4] = human[12]
        new_human[5] = human[13]
        new_human[6] = human[4]
        new_human[7] = human[3]
        new_human[8] = human[2]
        new_human[9] = human[5]
        new_human[10] = human[6]
        new_human[11] = human[7]
        new_human[12] = human[1]

        top_head = None
        nose = human[0]
        eyer = human[14]
        eyel = human[15]
        earr = human[16]
        earl = human[17]
        top_head_items = [elem for elem in [nose, eyel, eyer, earr, earl]
                          if elem is not None]
        if len(top_head_items) > 0:
            top_head_items = np.array(top_head_items)
            assert len(top_head_items.shape) == 2
            top_head = np.mean(top_head_items, axis=0)
        new_human[13] = top_head
        human_t.append(new_human)

    return human_t