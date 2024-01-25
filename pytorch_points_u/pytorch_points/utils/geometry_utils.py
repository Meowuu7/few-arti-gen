"""
Courtesy to
https://github.com/ranahanocka/MeshCNN/blob/master/models/layers/mesh_prepare.py
"""
import numpy as np
import random
import openmesh as om
import os
from matplotlib import cm
import torch
from collections import abc
from ..network.geo_operations import compute_face_normals_and_areas
from ..misc import logger

def normalize_to_same_area(v_ref: torch.Tensor, f_ref: torch.Tensor, v: torch.Tensor, f:torch.Tensor):
    """
    normalize mesh(v,f) to have the same surface area as mesh(v_ref, f_ref)
    """
    area_ref = compute_face_normals_and_areas(v_ref,f_ref)[1]
    area = compute_face_normals_and_areas(v, f)[1]
    area_ref = torch.sum(area_ref, dim=-1)
    area = torch.sum(area, dim=-1)
    ratio = torch.sqrt(area_ref/area).unsqueeze(-1).unsqueeze(-1)
    v = v*ratio
    return v


def read_trimesh(filename, return_mesh=False, **kwargs):
    """
    load vertices and faces of a mesh file
    return:
        V (N,3) floats
        F (F,3) int64
        properties "vertex_colors" "face_colors" etc
        mesh    trimesh object
    """
    try:
        mesh = om.read_trimesh(filename, **kwargs)
    except RuntimeError as e:
        mesh = om.read_trimesh(filename)

    V = mesh.points()
    face_lists = []
    for f in mesh.face_vertex_indices():
        face_lists.append(f)
    F = np.stack(face_lists, axis=0)

    mesh.request_face_normals()
    if not mesh.has_vertex_normals():
        mesh.request_vertex_normals()
        mesh.update_normals()
    v_normals = mesh.vertex_normals()
    f_normals = mesh.face_normals()
    V = np.concatenate([V, v_normals], axis=-1)
    F = np.concatenate([F, f_normals], axis=-1)

    properties = {}
    if mesh.has_vertex_colors():
        v_colors = mesh.vertex_colors()
        properties["vertex_colors"] = v_colors
    if mesh.has_face_colors():
        f_colors = mesh.face_colors()
        properties["face_colors"] = f_colors

    if return_mesh:
        return V, F, properties, mesh

    return V, F, properties

def write_trimesh(filename, V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1", **kwargs):
    """
    write a mesh with (N,3) vertices and (F,3) faces to file
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)

    mesh = array_to_mesh(V, F, v_colors=v_colors, f_colors=f_colors, v_normals=v_normals, cmap_name=cmap_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    om.write_mesh(filename, mesh, vertex_color=mesh.has_vertex_colors(), **kwargs)



def array_to_mesh(V, F, v_colors=None, f_colors=None, v_normals=True, cmap_name="Set1"):
    """
    convert a mesh with (N,3) vertices and (F,3) faces to a trimesh object
    """
    assert(V.ndim==2)
    assert(F.ndim==2)
    assert(F.shape[-1]==3)
    if isinstance(V, torch.Tensor):
        V = V.detach().cpu().numpy()
    if isinstance(F, torch.Tensor):
        F = F.detach().cpu().numpy()

    mesh = om.TriMesh(points=V, face_vertex_indices=F)
    if v_colors is not None:
        if isinstance(v_colors, torch.Tensor):
            v_colors = v_colors.detach().cpu().numpy()
        assert(v_colors.shape[0]==V.shape[0])
        # 1D scalar for each face
        if v_colors.size == V.shape[0]:
            cmap = cm.get_cmap(cmap_name)
            minV, maxV = v_colors.min(), v_colors.max()
            v_colors = (v_colors-minV)/maxV
            v_colors = [cmap(color) for color in v_colors]
        else:
            assert(v_colors.shape[1]==3 or v_colors.shape[1]==4)
            if v_colors.shape[1] == 3:
                mesh.vertex_colors()[:,-1]=1
        mesh.request_vertex_colors()
        mesh.vertex_colors()[:] = v_colors

    if f_colors is not None:
        assert(f_colors.shape[0]==F.shape[0])
        # 1D scalar for each face
        if f_colors.size == F.shape[0]:
            cmap = cm.get_cmap(cmap_name)
            minV, maxV = f_colors.min(), f_colors.max()
            f_colors = (f_colors-minV)/maxV
            f_colors = [cmap(color) for color in f_colors]
        else:
            assert(f_colors.shape[1]==3 or f_colors.shape[1]==4)
            if f_colors.shape[1] == 3:
                mesh.face_colors()[:,-1]=1
        mesh.request_face_colors()
        mesh.face_colors()[:] = f_colors

    mesh.request_face_normals()
    if v_normals:
        mesh.request_vertex_normals()

    mesh.update_normals()
    return mesh


def generatePolygon( ctrX, ctrY, aveRadius, irregularity, spikeyness, randRot, numVerts) :
    '''Start with the centre of the polygon at ctrX, ctrY,
    then creates the polygon by sampling points on a circle around the centre.
    Randon noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    Params:
    ctrX, ctrY - coordinates of the "centre" of the polygon, scalars
    aveRadius - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    irregularity - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    spikeyness - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, aveRadius]
    randRot - [0, 1] indicating how much variance there is in the mean angular position of the first vertex
    numVerts - self-explanatory

    Returns a list of vertices, in CCW order.
    '''

    irregularity = np.clip( irregularity, 0,1 ) * 2*np.pi / numVerts
    spikeyness = np.clip( spikeyness, 0,1 ) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2*np.pi / numVerts) - irregularity
    upper = (2*np.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts) :
        tmp = np.random.uniform(lower, upper)
        angleSteps.append( tmp )
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2*np.pi)
    for i in range(numVerts) :
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    # angle = random.uniform(0, 2*np.pi)
    angle = random.uniform(0, randRot)*2*np.pi
    for i in range(numVerts) :
        r_i = np.clip(np.random.normal(aveRadius, spikeyness), 0, 2*aveRadius )
        x = ctrX + r_i*np.cos(angle)
        y = ctrY + r_i*np.sin(angle)
        points.append((x,y))

        angle = angle + angleSteps[i]

    return points

class Mesh(abc.Mapping):
    """
    create mesh object from vertices and faces with attributes
    ve:            List(List(int64)) vertex - edge idx
    edges:         (E,2) int32 numpy array edges represented as sorted vertex indices
    gemm_edges:     (E,4) int64 numpy array indices of the four neighboring edges
    ======
    :param
        vertices   (V,3) float32
        faces      (F,3) int64
    """
    def __init__(self, filepath: str = None, vertices: torch.Tensor = None, faces: torch.Tensor = None):
        self.vs = vertices
        self.fs = faces
        self.vn = None
        self.fn = None

        if (self.vs is not None and self.fs is not None):
            if filepath is not None:
                logger.warn("Using provided vertices and faces, ignore filepath")
            assert(self.vs.ndim == 2 and self.fs.ndim ==2)
            assert(self.vs.shape[-1] == 3)
            assert(self.fs.shape[-1] == 3)
        elif filepath is not None:
            mesh = om.read_trimesh(filepath)
            self.vs = torch.from_numpy(mesh.points()).to(dtype=torch.float32)

            face_lists = []
            for f in mesh.face_vertex_indices():
                face_lists.append(f)
            self.fs = torch.from_numpy(np.stack(face_lists, axis=0)).to(dtype=torch.int64)
            mesh.request_face_normals()
            if not mesh.has_vertex_normals():
                mesh.request_vertex_normals()
                mesh.update_normals()
            v_normals = mesh.vertex_normals()
            f_normals = mesh.face_normals()
            self.vn = torch.from_numpy(v_normals.astype(np.float32))
            self.fn = torch.from_numpy(f_normals.astype(np.float32))
        else:
            logger.error("[{}] Must provide a mesh".format(__class__))

        # build_gemm(self, self.fs)
        self.farea = compute_face_normals_and_areas(self.vs, self.fs)[1]
        self.features = ['vs', 'fs', 'vn', 'fn', 'farea']

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        return iter(self.features)


def build_gemm(mesh, faces):
    """
    ve:            List(List(int64)) vertex - edge idx
    edges:         (E,2) int32 numpy array edges represented as sorted vertex indices
    gemm_edges     (E,4) int64 numpy array indices of the four neighboring edges
    """
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    mesh.ve = [[] for _ in mesh.vs]
    edge_nb = []
    # sides = []
    edge2key = dict()
    edges = []
    edges_count = 0
    nb_count = []
    for face_id, face in enumerate(faces):
        faces_edges = []
        for i in range(3):
            cur_edge = (face[i], face[(i + 1) % 3])
            faces_edges.append(cur_edge)

        for idx, edge in enumerate(faces_edges):
            edge = tuple(sorted(list(edge)))
            faces_edges[idx] = edge
            if edge not in edge2key:
                edge2key[edge] = edges_count
                edges.append(list(edge))
                edge_nb.append([-1, -1, -1, -1])
                # sides.append([-1, -1, -1, -1])
                mesh.ve[edge[0]].append(edges_count)
                mesh.ve[edge[1]].append(edges_count)
                # mesh.edge_areas.append(0)
                nb_count.append(0)
                edges_count += 1
            # mesh.edge_areas[edge2key[edge]] += face_areas[face_id] / 3
        for idx, edge in enumerate(faces_edges):
            edge_key = edge2key[edge]
            edge_nb[edge_key][nb_count[edge_key]]     = edge2key[faces_edges[(idx + 1) % 3]]
            edge_nb[edge_key][nb_count[edge_key] + 1] = edge2key[faces_edges[(idx + 2) % 3]]
            nb_count[edge_key] += 2
        # for idx, edge in enumerate(faces_edges):
        #     edge_key = edge2key[edge]
        #     sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[faces_edges[(idx + 1) % 3]]] - 1
        #     sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[faces_edges[(idx + 2) % 3]]] - 2
    mesh.edges = np.array(edges, dtype=np.int32)
    mesh.gemm_edges = np.array(edge_nb, dtype=np.int64)
    # mesh.sides = np.array(sides, dtype=np.int64
    mesh.edges_count = edges_count
    # mesh.edge_areas = vertices(mesh.edge_areas, dtype=np.float32) / np.sum(face_areas) #todo whats the difference between edge_areas and edge_lengths?


def get_edge_points(mesh):
    """
    get 4 edge points of all points
    ===
    return:
        edge_points (E, 4) int64
    """
    edge_points = np.zeros([mesh.edges_count, 4], dtype=np.int64)
    for edge_id, edge in enumerate(mesh.edges):
        edge_points[edge_id] = get_side_points(mesh, edge_id)
    return edge_points


def get_side_points(mesh, edge_id):
    """
    return the four vertex indices around an edge
         1
    e/d /|\ c
       / | \
    3 /  |a \ 2
      \  |  /
       \ | /
    d/e \|/ b
         0
    """
    edge_a = mesh.edges[edge_id]

    if mesh.gemm_edges[edge_id, 0] == -1:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    else:
        edge_b = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_c = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    if mesh.gemm_edges[edge_id, 2] == -1:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 0]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 1]]
    else:
        edge_d = mesh.edges[mesh.gemm_edges[edge_id, 2]]
        edge_e = mesh.edges[mesh.gemm_edges[edge_id, 3]]
    first_vertex = 0
    second_vertex = 0
    third_vertex = 0
    if edge_a[1] in edge_b:
        first_vertex = 1
    if edge_b[1] in edge_c:
        second_vertex = 1
    if edge_d[1] in edge_e:
        third_vertex = 1
    return [edge_a[first_vertex], edge_a[1 - first_vertex], edge_b[second_vertex], edge_d[third_vertex]]
