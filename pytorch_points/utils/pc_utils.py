"""
Utility functions for processing point clouds.
"""
import os
import numpy as np
import torch
# Point cloud IO
from matplotlib import cm
import matplotlib.colors as mpc
import plyfile


def normalize_to_sphere(input):
    """
    recenter point cloud to mean value and rescale to fit inside a unit ball
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    if isinstance(input, np.ndarray):
        centroid = np.mean(input, axis=axis, keepdims=True)
        input = input - centroid
        furthest_distance = np.amax(
            np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        centroid = torch.mean(input, dim=axis, keepdims=True)
        input = input - centroid
        furthest_distance = torch.max(
            torch.sqrt(torch.sum(input ** 2, dim=-1, keepdims=True)), dim=axis, keepdims=True)[0]
        input = input / furthest_distance

    return input, centroid, furthest_distance

def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP+minP)/2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP+minP)/2
        input = input - centroid
        in_shape = list(input.shape[:axis])+[P*D]
        furthest_distance = torch.max(torch.abs(input).view(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance

    return input, centroid, furthest_distance

def center_bounding_box(points):
    """
    centers the shapes
    Args:
        points: (*,N,D) D-dimensional points
    Outputs:
        points: centerred points
        centroid: (*,D)
        bbox:     (*,D)
    """
    is_torch = isinstance(points, torch.Tensor)
    if is_torch:
        device = points.device
        points = points.cpu().numpy()

    min_vals = np.min(points, -2, keepdims=True)
    max_vals = np.max(points, -2, keepdims=True)
    centroid = (min_vals + max_vals) / 2
    points = points - centroid
    bbox = (max_vals - min_vals)/2
    if is_torch:
        points = torch.from_numpy(points).to(device=device)
        centroid = torch.from_numpy(centroid).to(device=device)
        bbox = torch.from_numpy(bbox).to(device=device)
    return points, centroid, bbox


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02, is_2D=False):
    """
    Randomly jitter points. jittering is per point.
    Input:
        batch_data: BxNx3 array, original batch of point clouds
    Return:
        BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    chn = 2 if is_2D else 3
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -clip, clip).astype(dtype=batch_data.dtype)
    jittered_data[:, :, chn:] = 0
    jittered_data += batch_data
    return jittered_data


def rotate_point_cloud_and_gt(batch_data, batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=batch_data.dtype)
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=batch_data.dtype)
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]], dtype=batch_data.dtype)
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        batch_data[k, ..., 0:3] = np.dot(
            batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(
                batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3] = np.dot(
                batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(
                    batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data, batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud, i.e. isotropic
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, (B, 1, 1)).astype(batch_data.dtype)

    batch_data = np.concatenate([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

    if batch_gt is not None:
        batch_gt = np.concatenate([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

    return batch_data, batch_gt, np.squeeze(scales)


def downsample_points(pts, K):
    # if num_pts > 2K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2 * K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K,
                                    replace=False), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        farthest_pts = np.zeros((k, pts.shape[1]), dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3]))
        return farthest_pts


def read_ply_with_color(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)
    colors = None
    if 'red' in loaded['vertex'].data.dtype.names:
        colors = np.vstack([loaded['vertex'].data['red'],
                            loaded['vertex'].data['green'], loaded['vertex'].data['blue']])
        if 'alpha' in loaded['vertex'].data.dtype.names:
            colors = np.concatenate([colors, np.expand_dims(
                loaded['vertex'].data['alpha'], axis=0)], axis=0)
        colors = colors.transpose(1, 0)
        colors = colors.astype(np.float32) / 255.0

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points, colors


def read_ply(file, count=None):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)

    points = points.transpose(1, 0)
    if count is not None:
        if count > points.shape[0]:
            # fill the point clouds with the random point
            tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
            tmp[:points.shape[0], ...] = points
            tmp[points.shape[0]:, ...] = points[np.random.choice(
                points.shape[0], count - points.shape[0]), :]
            points = tmp
        elif count < points.shape[0]:
            # different to pointnet2, take random x point instead of the first
            # idx = np.random.permutation(count)
            # points = points[idx, :]
            points = downsample_points(points, count)
    return points

def read_ply_with_face(file):
    loaded = plyfile.PlyData.read(file)
    points = np.vstack([loaded['vertex'].data['x'],
                        loaded['vertex'].data['y'], loaded['vertex'].data['z']])
    if 'nx' in loaded['vertex'].data.dtype.names:
        normals = np.vstack([loaded['vertex'].data['nx'],
                             loaded['vertex'].data['ny'], loaded['vertex'].data['nz']])
        points = np.concatenate([points, normals], axis=0)

    points = points.transpose(1, 0)
    faces = np.vstack([loaded["face"].data[i][0] for i in range(loaded["face"].count)])
    return points, faces


def save_ply_with_face_property(filename, points, faces, property, property_max, cmap_name="Set1", binary=True):
    face_num = faces.shape[0]
    colors = np.full(faces.shape, 0.5)
    cmap = cm.get_cmap(cmap_name)
    for point_idx in range(face_num):
        colors[point_idx] = cmap(property[point_idx] / property_max)[:3]
    save_ply_with_face(filename, points, faces, colors, binary=True)


def save_ply_with_face(faces, filename, points, colors=None, binary=True):
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)
    vertex = np.array([tuple(p) for p in points], dtype=[
                      ('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces = np.array([(tuple(p),) for p in faces], dtype=[
                     ('vertex_indices', 'i4', (len(faces[0]), ))])
    descr = faces.dtype.descr
    if colors is not None:
        assert len(colors) == len(faces)
        face_colors = np.array([tuple(c * 255) for c in colors],
                               dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        descr = faces.dtype.descr + face_colors.dtype.descr

    faces_all = np.empty(len(faces), dtype=descr)
    for prop in faces.dtype.names:
        faces_all[prop] = faces[prop]
    if colors is not None:
        for prop in face_colors.dtype.names:
            faces_all[prop] = face_colors[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(
        vertex, 'vertex'), plyfile.PlyElement.describe(faces_all, 'face')], text=(not binary))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def load(filename, count=None):
    if filename[-4:] == ".ply":
        points = read_ply(filename, count).astype(np.float32)
    else:
        points = np.loadtxt(filename).astype(np.float32)
        if count is not None:
            if count > points.shape[0]:
                # fill the point clouds with the random point
                tmp = np.zeros((count, points.shape[1]), dtype=points.dtype)
                tmp[:points.shape[0], ...] = points
                tmp[points.shape[0]:, ...] = points[np.random.choice(
                    points.shape[0], count - points.shape[0]), :]
                points = tmp
            elif count < points.shape[0]:
                # different to pointnet2, take random x point instead of the first
                # idx = np.random.permutation(count)
                # points = points[idx, :]
                points = downsample_points(points, count)
    return points


def save_ply(filename, points, colors=None, normals=None, binary=True):
    """
    save 3D/2D points to ply file
    Args:
        points (numpy array): (N,2or3)
        colors (numpy uint8 array): (N, 3or4)
    """
    assert(points.ndim == 2)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)

    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        assert(normals.ndim == 2)
        if normals.shape[-1] == 2:
            normals = np.concatenate([normals, np.zeros_like(normals)[:, :1]], axis=-1)
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=(not binary))
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def save_ply_property(filename, points, property, property_max=None, property_min=None, normals=None, cmap_name='Set1', binary=True):
    point_num = points.shape[0]
    colors = np.full([point_num, 3], 0.5)
    cmap = cm.get_cmap(cmap_name)
    if property_max is None:
        property_max = np.amax(property, axis=0)
    if property_min is None:
        property_min = np.amin(property, axis=0)
    p_range = property_max-property_min
    if property_max == property_min:
        property_max = property_min+1
    normalizer = mpc.Normalize(vmin=property_min, vmax=property_max)
    p = normalizer(property)
    colors = cmap(p)[:,:3]
    save_ply(filename, points, colors, normals, binary)

def save_pts(filename, points, normals=None, labels=None):
    assert(points.ndim==2)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros_like(points)[:, :1]], axis=-1)
    if normals is not None:
        points = np.concatenate([points, normals], axis=1)
    if labels is not None:
        points = np.concatenate([points, labels], axis=1)
        np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1]+["\"%i\""])
    else:
        np.savetxt(filename, points, fmt=["%.10e"]*points.shape[1])

"""
augmentation operations for a point cloud (TODO: extend to batches of point clouds)
https://github.com/ThibaultGROUEIX/CycleConsistentDeformation/blob/master/auxiliary/normalize_points.py
"""
def get_3D_rot_matrix(axis, angle):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    if axis == 1:
        return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [- np.sin(angle), 0, np.cos(angle)]])
    if axis == 2:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [1, 0, 0]])

def uniform_rotation_axis_matrix(axis=0, range_rot=360):
    # input : Numpy Tensor N_pts, 3
    # ouput : Numpy Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation around axis
    scale_factor = 360.0 / range_rot
    theta = np.random.uniform(- np.pi / scale_factor, np.pi / scale_factor)
    rot_matrix = get_3D_rot_matrix(axis, theta)
    return torch.from_numpy(np.transpose(rot_matrix)).float()


def uniform_rotation_axis(points, axis=0, normals=False, range_rot=360):
    # input : Numpy Tensor N_pts, 3
    # ouput : Numpy Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation around axis
    rot_matrix = uniform_rotation_axis_matrix(axis, range_rot)

    if isinstance(points, torch.Tensor):
        points[:, :3] = torch.mm(points[:, :3], rot_matrix)
        if normals:
            points[:, 3:6] = torch.mm(points[:, 3:6], rot_matrix)
        return points, rot_matrix
    elif isinstance(points, np.ndarray):
        points = points.copy()
        points[:, :3] = points[:, :3].dot(rot_matrix.numpy())
        if normals:
            points[:, 3:6] = points[:, 3:6].dot(rot_matrix.numpy())
        return points, rot_matrix
    else:
        print("Pierre-Alain was right.")

def anisotropic_scaling(points):
    # input : points : N_point, 3
    scale = torch.rand(1, 3) / 2.0 + 0.75  # uniform sampling 0.75, 1.25
    return scale * points  # Element_wize multiplication with broadcasting

def uniform_rotation_sphere(points, normals=False):
    # input : Tensor N_pts, 3
    # ouput : Tensor N_pts, 3
    # ouput : rot matrix Numpy Tensor 3, 3
    # Uniform random rotation on the sphere
    x = torch.Tensor(2)
    x.uniform_()
    p = torch.Tensor([[np.cos(np.pi * 2 * x[0]) * np.sqrt(x[1]),
                       (np.random.binomial(1, 0.5, 1)[0] * 2 - 1) * np.sqrt(1 - x[1]),
                       np.sin(np.pi * 2 * x[0]) * np.sqrt(x[1])]])
    z = torch.Tensor([[0, 1, 0]])
    v = (p - z) / (p - z).norm()
    H = torch.eye(3) - 2 * torch.matmul(v.transpose(1, 0), v)
    rot_matrix = - H

    if isinstance(points, torch.Tensor):
        points[:, :3] = torch.mm(points[:, :3], rot_matrix)
        if normals:
            points[:, 3:6] = torch.mm(points[:, 3:6], rot_matrix)
        return points, rot_matrix

    elif isinstance(points, np.ndarray):
        points[:, :3] = points[:, :3].dot(rot_matrix.numpy())
        if normals:
            points[:, 3:6] = points[:, 3:6].dot(rot_matrix.numpy())
        return points, rot_matrix
    else:
        print("Pierre-Alain was right.")

def add_random_translation(points, scale=0.03):
    # input : Numpy Tensor N_pts, D_dim
    # ouput : Numpy Tensor N_pts, D_dim
    # Uniform random translation on first 3 dimensions
    a = torch.FloatTensor(3)
    points[:, 0:3] = points[:, 0:3] + (a.uniform_(-1, 1) * scale).unsqueeze(0).expand(-1, 3)
    return points

def random_sphere(batch, num_points):
    """generate random samples on a uni-sphere (B,N,3)"""
    # double theta = 2 * M_PI * uniform01(generator);
    # double phi = M_PI * uniform01(generator);
    # double x = sin(phi) * cos(theta);
    # double y = sin(phi) * sin(theta);
    # double z = cos(phi);
    theta = np.random.rand(batch, num_points)*2*np.pi
    phi = np.random.rand(batch, num_points)*np.pi
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return np.stack([x,y,z], axis=-1)
