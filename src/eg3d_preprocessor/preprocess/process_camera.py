import numpy as np


def fix_intrinsics(intrinsics):
    intrinsics = np.array(intrinsics).copy()
    assert intrinsics.shape == (3, 3), intrinsics
    intrinsics[0, 0] = 2985.29 / 700
    intrinsics[1, 1] = 2985.29 / 700
    intrinsics[0, 2] = 1 / 2
    intrinsics[1, 2] = 1 / 2
    assert intrinsics[0, 1] == 0
    assert intrinsics[2, 2] == 1
    assert intrinsics[1, 0] == 0
    assert intrinsics[2, 0] == 0
    assert intrinsics[2, 1] == 0
    return intrinsics


def fix_pose(pose):
    COR = np.array([0, 0, 0.175])
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    direction = (location - COR) / np.linalg.norm(location - COR)
    pose[:3, 3] = direction * 2.7 + COR
    return pose


def fix_pose_orig(pose):
    pose = np.array(pose).copy()
    location = pose[:3, 3]
    radius = np.linalg.norm(location)
    pose[:3, 3] = pose[:3, 3] / radius * 2.7
    return pose


def flip_yaw(pose_matrix):
    flipped = pose_matrix.copy()
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped


def process_camera(pose, intrinsics):
    # if args.mode == 'cor':
    #     pose = fix_pose(pose)
    # elif args.mode == 'orig':
    pose = fix_pose_orig(pose)
    # else:
    #     assert False, "invalid mode"
    intrinsics = fix_intrinsics(intrinsics)
    label = np.concatenate([pose.reshape(-1), intrinsics.reshape(-1)])
    return label