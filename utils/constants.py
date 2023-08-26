from options.options import opt

THU_LAB_FLAG = "nas"
MEG_LAB_FLAG = "/data/datasets"


MASK_VERTICES_TYPE_COORD = "coord"
MASK_VERTICES_TYPE_POS = "pos"

MASK_COORD_VALUE = (2 ** opt.model.quantization_bits) // opt.vertex_model.grid_size
MASK_GRID_VALIE = 2 ** (opt.vertex_model.grid_size ** 3)

PART_SEP_GRID_VALUE = 2 ** (opt.vertex_model.grid_size ** 3) + 1
ENDING_GRID_VALUE = 2 ** (opt.vertex_model.grid_size ** 3) + 2

PART_SEP_XYZ = [-1, -1, 0]
ENDING_XYZ = [-1, -1, -1]