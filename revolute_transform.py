import numpy as np


def revoluteTransform(points, pivotPoint, axis, theta): 
  # theta: 0.5 * np.pi
  # points:B*n*3 or B*n*4 or n*3 or n*4
  # pivotPoint denotes pivot point, axis denotes rotation direction
  size = points.shape
  if len(size) == 3:
    if size[-1] > 4:
      points = points.transpose(0, 2, 1)
  elif size[-1] > 4:
    points = points.transpose(1, 0)
  size = points.shape
  n = size[-2]
  if size[-1] == 3:
    one = np.ones((n, 1)) if len(size) == 2 else np.ones((size[0], n, 1))
    points = np.concatenate((points, one), axis=-1)

  try:
    a = pivotPoint[0]
    b = pivotPoint[1]
    c = pivotPoint[2]
  except:
    a = pivotPoint[0][0]
    b = pivotPoint[0][1]
    c = pivotPoint[0][2]

  try:
    u = axis[0]
    v = axis[1]
    w = axis[2]
  except:
    u = axis[0][0]
    v = axis[0][1]
    w = axis[0][2]

  uu = u * u
  uv = u * v
  uw = u * w
  vv = v * v
  vw = v * w
  ww = w * w
  au = a * u
  av = a * v
  aw = a * w
  bu = b * u
  bv = b * v
  bw = b * w
  cu = c * u
  cv = c * v
  cw = c * w

  costheta = np.cos(theta)
  sintheta = np.sin(theta)

  m = np.zeros((4, 4))
  m[0, 0] = uu + (vv + ww) * costheta
  m[1, 0] = uv * (1 - costheta) + w * sintheta
  m[2, 0] = uw * (1 - costheta) - v * sintheta
  m[3, 0] = 0

  m[0, 1] = uv * (1 - costheta) - w * sintheta
  m[1, 1] = vv + (uu + ww) * costheta
  m[2, 1] = vw * (1 - costheta) + u * sintheta
  m[3, 1] = 0

  m[0, 2] = uw * (1 - costheta) + v * sintheta
  m[1, 2] = vw * (1 - costheta) - u * sintheta
  m[2, 2] = ww + (uu + vv) * costheta
  m[3, 2] = 0

  m[0, 3] = (a * (vv + ww) - u * (bv + cw)) * (1 - costheta) + (bw - cv) * sintheta
  m[1, 3] = (b * (uu + ww) - v * (au + cw)) * (1 - costheta) + (cu - aw) * sintheta
  m[2, 3] = (c * (uu + vv) - w * (au + bv)) * (1 - costheta) + (av - bu) * sintheta
  m[3, 3] = 1
  m = m.transpose(1, 0)

  points = np.dot(points, m)
  return points, m