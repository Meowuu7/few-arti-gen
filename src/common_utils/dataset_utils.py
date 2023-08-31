import torch
import torch.nn as nn 
import numpy as np


def get_edges_from_faces(vertices, faces):
  ### n_faces x 3 ###
  edges = []
  dofs = np.zeros((vertices.shape[0], ), dtype=np.float32)
  
  for i_f in range(faces.shape[0]):
    cur_f = faces[i_f].tolist()
    for i0, v0 in enumerate(cur_f):
      i1 = (i0 + 1) % len(cur_f)
      v1 = cur_f[i1]
      edges += [[v0, v1], [v1, v0]]
      dofs[v0 - 1] += 1
  edges = np.array(edges, dtype=np.long) # n_edge_pairs x 2
  edges = np.transpose(edges, (1, 0))
  return edges, dofs
    

