"""Mesh data utilities."""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # pylint: disable=unused-import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import networkx as nx
import numpy as np
import six
import os

def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)

def read_obj_file(obj_file):
  """Read vertices and faces from already opened file."""
  vertex_list = []
  flat_vertices_list = []
  flat_vertices_indices = {}
  flat_triangles = []

  for line in obj_file:
    tokens = line.split()
    if not tokens:
      continue
    line_type = tokens[0]
    # We skip lines not starting with v or f.
    if line_type == 'v': # 
      vertex_list.append([float(x) for x in tokens[1:]])
    elif line_type == 'f':
      triangle = []
      for i in range(len(tokens) - 1):
        vertex_name = tokens[i + 1]
        if vertex_name in flat_vertices_indices: # triangles
          triangle.append(flat_vertices_indices[vertex_name])
          continue
        flat_vertex = []
        for index in six.ensure_str(vertex_name).split('/'):
          if not index:
            continue
          # obj triangle indices are 1 indexed, so subtract 1 here.
          flat_vertex += vertex_list[int(index) - 1]
        flat_vertex_index = len(flat_vertices_list)
        flat_vertices_list.append(flat_vertex)
        # flat_vertex_index
        flat_vertices_indices[vertex_name] = flat_vertex_index
        triangle.append(flat_vertex_index)
      flat_triangles.append(triangle)

  return np.array(flat_vertices_list, dtype=np.float32), flat_triangles


def read_obj(obj_path):
  """Open .obj file from the path provided and read vertices and faces."""

  with open(obj_path) as obj_file:
    return read_obj_file(obj_file)



def center_vertices(vertices):
  """Translate the vertices so that bounding box is centered at zero."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  vert_center = 0.5 * (vert_min + vert_max)
  return vertices - vert_center


def normalize_vertices_scale(vertices):
  """Scale the vertices so that the long diagonal of the bounding box is one."""
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  extents = vert_max - vert_min
  scale = np.sqrt(np.sum(extents**2))
  return vertices / scale


def quantize_verts(verts, n_bits=8):
  """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts_quantize = (verts - min_range) * range_quantize / (
      max_range - min_range)
  return verts_quantize.astype('int32')


def dequantize_verts(verts, n_bits=8, add_noise=False):
  """Convert quantized vertices to floats."""
  min_range = -0.5
  max_range = 0.5
  range_quantize = 2**n_bits - 1
  verts = verts.astype('float32')
  verts = verts * (max_range - min_range) / range_quantize + min_range
  if add_noise:
    verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
  return verts


### dump vertices and faces to the obj file
def write_obj(vertices, faces, file_path, transpose=True, scale=1.):
  """Write vertices and faces to obj."""
  if transpose:
    vertices = vertices[:, [1, 2, 0]]
  vertices *= scale
  if faces is not None:
    if min(min(faces)) == 0:
      f_add = 1
    else:
      f_add = 0
  with open(file_path, 'w') as f:
    for v in vertices:
      f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
    for face in faces:
      line = 'f'
      for i in face:
        line += ' {}'.format(i + f_add)
      line += '\n'
      f.write(line)


def face_to_cycles(face):
  """Find cycles in face."""
  g = nx.Graph()
  for v in range(len(face) - 1):
    g.add_edge(face[v], face[v + 1])
  g.add_edge(face[-1], face[0])
  return list(nx.cycle_basis(g))


def flatten_faces(faces):
  """Converts from list of faces to flat face array with stopping indices."""
  if not faces:
    return np.array([0])
  else:
    l = [f + [-1] for f in faces[:-1]]
    l += [faces[-1] + [-2]]
    return np.array([item for sublist in l for item in sublist]) + 2  # pylint: disable=g-complex-comprehension


def unflatten_faces(flat_faces):
  """Converts from flat face sequence to a list of separate faces."""
  def group(seq):
    g = []
    for el in seq:
      if el == 0 or el == -1:
        yield g
        g = []
      else:
        g.append(el - 1)
    yield g
  outputs = list(group(flat_faces - 1))[:-1]
  # Remove empty faces
  return [o for o in outputs if len(o) > 2]



def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
  """Quantize vertices, remove resulting duplicates and reindex faces."""
  vertices = quantize_verts(vertices, quantization_bits)
  vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

  # Sort vertices by z then y then x.
  sort_inds = np.lexsort(vertices.T)
  vertices = vertices[sort_inds]

  # Re-index faces and tris to re-ordered vertices.
  faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
  if tris is not None:
    tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

  # Merging duplicate vertices and re-indexing the faces causes some faces to
  # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
  # sub-faces.
  sub_faces = []
  for f in faces:
    cliques = face_to_cycles(f)
    for c in cliques:
      c_length = len(c)
      # Only append faces with more than two verts.
      if c_length > 2:
        d = np.argmin(c)
        # Cyclically permute faces just that first index is the smallest.
        sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
  faces = sub_faces
  if tris is not None:
    tris = np.array([v for v in tris if len(set(v)) == len(v)])

  # Sort faces by lowest vertex indices. If two faces have the same lowest
  # index then sort by next lowest and so on.
  faces.sort(key=lambda f: tuple(sorted(f)))
  if tris is not None:
    tris = tris.tolist()
    tris.sort(key=lambda f: tuple(sorted(f)))
    tris = np.array(tris)

  # After removing degenerate faces some vertices are now unreferenced.
  # Remove these.
  num_verts = vertices.shape[0]
  vert_connected = np.equal(
      np.arange(num_verts)[:, None], np.hstack(faces)[None]).any(axis=-1)
  vertices = vertices[vert_connected]

  # Re-index faces and tris to re-ordered vertices.
  vert_indices = (
      np.arange(num_verts) - np.cumsum(1 - vert_connected.astype('int')))
  faces = [vert_indices[f].tolist() for f in faces]
  if tris is not None:
    tris = np.array([vert_indices[t].tolist() for t in tris])

  return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8, recenter_mesh=True):
  """Process mesh vertices and faces."""

  # Transpose so that z-axis is vertical.
  vertices = vertices[:, [2, 0, 1]]

  if recenter_mesh:
    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

  # Quantize and sort vertices, remove resulting duplicates, sort and reindex
  # faces.
  vertices, faces, _ = quantize_process_mesh(
      vertices, faces, quantization_bits=quantization_bits)

  # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
  faces = flatten_faces(faces)

  # Discard degenerate meshes without faces.
  return {
      'vertices': vertices,
      'faces': faces,
  }



def plot_sampled_meshes(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample['left'], v_sample['rgt']]
  part_face_samples = [f_sample['left'], f_sample['rgt']]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  if predict_joint:
    pred_dir = v_sample['joint_dir']
    pred_pvp = v_sample['joint_pvp']
    print("pred_dir", pred_dir.shape, pred_dir)
    print("pred_pvp", pred_pvp.shape, pred_pvp)
  else:
    pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  
  


  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)
      # and write this obj file?
      # write_obj(vertices, faces, file_path, transpose=True, scale=1.):
      # write mesh objs
      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
        
      

  ###### plot mesh (predicted) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}.obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
  ###### plot mesh (predicted) ######


  ###### plot mesh (translated) ######
  tot_samples_mesh_dict = []
  for i_s in range(tot_n_samples):
      cur_s_tot_vertices = []
      cur_s_tot_faces = []
      cur_s_n_vertices = 0
      cur_s_pred_pvp = pred_pvp[i_s]
      
      for i_p in range(tot_n_part):
          cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
          cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                          cur_s_cur_part_mesh_dict['faces']
          cur_s_cur_part_new_faces = []
          for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
              cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
              cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
          cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]

          if i_p == 1:
            # min_rngs = cur_s_cur_part_vertices.min(1)
            # max_rngs = cur_s_cur_part_vertices.max(1)
            min_rngs = cur_s_cur_part_vertices.min(0)
            max_rngs = cur_s_cur_part_vertices.max(0)
            # shifted; cur_s_pred_pvp
            # shifted = np.array([0., cur_s_pred_pvp[1] - max_rngs[1], cur_s_pred_pvp[2] - min_rngs[2]], dtype=np.float)
            # shifted = np.reshape(shifted, [1, 3]) # 
            cur_s_pred_pvp = np.array([0., max_rngs[1], min_rngs[2]], dtype=np.float32)
            pvp_sample_pred_err = np.sum((cur_s_pred_pvp - pred_pvp[i_s]) ** 2)
            # print prediction err, pred pvp and real pvp
            # print("cur_s, sample_pred_pvp_err:", pvp_sample_pred_err.item(), ";real val:", cur_s_pred_pvp, "; pred_val:", pred_pvp[i_s])
            pred_pvp[i_s] = cur_s_pred_pvp
            shifted = np.zeros((1, 3), dtype=np.float32)
            cur_s_cur_part_vertices = cur_s_cur_part_vertices + shifted # shift vertices... # min_rngs
          # shifted
          cur_s_tot_vertices.append(cur_s_cur_part_vertices)
          cur_s_tot_faces += cur_s_cur_part_new_faces

      cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
      cur_s_mesh_dict = {
          'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
      }
      tot_samples_mesh_dict.append(cur_s_mesh_dict)

  for i_s in range(tot_n_samples):
    cur_mesh = tot_samples_mesh_dict[i_s]
    cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
    cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_shifted.obj")
    if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
      write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)
  ###### plot mesh (translated) ######



  ###### plot mesh (rotated) ######
  if predict_joint:
    from revolute_transform import revoluteTransform
    tot_samples_mesh_dict = []
    for i_s in range(tot_n_samples):
        cur_s_tot_vertices = []
        cur_s_tot_faces = []
        cur_s_n_vertices = 0
      
        # cur_s_pred_dir = pred_dir[i_s]
        cur_s_pred_pvp = pred_pvp[i_s]
        print("current pred dir:", cur_s_pred_dir, "; current pred pvp:", cur_s_pred_pvp)
        cur_s_pred_dir = np.array([1.0, 0.0, 0.0], dtype=np.float)
        # cur_s_pred_pvp = cur_s_pred_pvp[[1, 2, 0]]

        for i_p in range(tot_n_part):
            cur_s_cur_part_mesh_dict = tot_mesh_list[i_p][i_s]
            cur_s_cur_part_vertices, cur_s_cur_part_faces = cur_s_cur_part_mesh_dict['vertices'], \
                                                            cur_s_cur_part_mesh_dict['faces']
                      
            if i_p == 1:
              cur_s_cur_part_vertices, _ = revoluteTransform(cur_s_cur_part_vertices, cur_s_pred_pvp, cur_s_pred_dir, 0.5 * np.pi) # reverse revolute vertices of the upper piece
              cur_s_cur_part_vertices = cur_s_cur_part_vertices[:, :3] # 
            cur_s_cur_part_new_faces = []
            for cur_s_cur_part_cur_face in cur_s_cur_part_faces:
                cur_s_cur_part_cur_new_face = [fid + cur_s_n_vertices for fid in cur_s_cur_part_cur_face]
                cur_s_cur_part_new_faces.append(cur_s_cur_part_cur_new_face)
            cur_s_n_vertices += cur_s_cur_part_vertices.shape[0]
            cur_s_tot_vertices.append(cur_s_cur_part_vertices)
            # print(f"i_s: {i_s}, i_p: {i_p}, n_vertices: {cur_s_cur_part_vertices.shape[0]}")
            cur_s_tot_faces += cur_s_cur_part_new_faces

        cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
        # print(f"i_s: {i_s}, n_cur_s_tot_vertices: {cur_s_tot_vertices.shape[0]}")
        cur_s_mesh_dict = {
            'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
        }
        tot_samples_mesh_dict.append(cur_s_mesh_dict)
    # plot_meshes(tot_samples_mesh_dict, ax_lims=0.5, mesh_sv_fn=f"./figs/training_step_{n}_part_{tot_n_part}_rot.png")  # plot the mesh;
    for i_s in range(tot_n_samples):
      cur_mesh = tot_samples_mesh_dict[i_s]
      cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
      # rotated mesh fn
      cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_ins_{i_s}_rot.obj")
      # write object in the file...
      if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
        write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=False, scale=1.)



def plot_sampled_meshes_single_part(v_sample, f_sample, sv_mesh_folder, cur_step=0, predict_joint=True,):
  
  if not os.path.exists(sv_mesh_folder):
    os.mkdir(sv_mesh_folder)

  part_vertex_samples = [v_sample]
  part_face_samples = [f_sample]


  tot_n_samples = part_vertex_samples[0]['vertices'].shape[0]
  tot_n_part = 2

  # not predict joints here
  # if predict_joint:
  #   pred_dir = v_sample['joint_dir']
  #   pred_pvp = v_sample['joint_pvp']
  #   print("pred_dir", pred_dir.shape, pred_dir)
  #   print("pred_pvp", pred_pvp.shape, pred_pvp)
  # else:
  #   pred_pvp = np.zeros(shape=[tot_n_samples, 3], dtype=np.float32)
  

  tot_mesh_list = []
  for i_p, (cur_part_v_samples_np, cur_part_f_samples_np) in enumerate(zip(part_vertex_samples, part_face_samples)):
      mesh_list = []
      for i_n in range(tot_n_samples):
          mesh_list.append(
              {
                  'vertices': cur_part_v_samples_np['vertices'][i_n][:cur_part_v_samples_np['num_vertices'][i_n]],
                  'faces': unflatten_faces(
                      cur_part_f_samples_np['faces'][i_n][:cur_part_f_samples_np['num_face_indices'][i_n]])
              }
          )
      tot_mesh_list.append(mesh_list)

      for i_n in range(tot_n_samples):
        cur_mesh = mesh_list[i_n]
        cur_mesh_vertices, cur_mesh_faces = cur_mesh['vertices'], cur_mesh['faces']
        # cur_mesh_sv_fn = os.path.join("./meshes", f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        cur_mesh_sv_fn = os.path.join(sv_mesh_folder, f"training_step_{cur_step}_part_{i_p}_ins_{i_n}.obj")
        print(f"saving to {cur_mesh_sv_fn}, nn_verts: {cur_mesh_vertices.shape[0]}, nn_faces: {len(cur_mesh_faces)}")
        if cur_mesh_vertices.shape[0] > 0 and len(cur_mesh_faces) > 0:
          write_obj(cur_mesh_vertices, cur_mesh_faces, cur_mesh_sv_fn, transpose=True, scale=1.)
        
      
