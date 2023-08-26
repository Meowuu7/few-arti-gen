from . import HierarchyArgmentParser

parser = HierarchyArgmentParser()



# Experiment arguments
dataset_args = parser.add_parser("dataset")
dataset_args.add_argument('--n-scales', type=int, default=1,
                          help='dataset tot use')

dataset_args.add_argument('--vox-size', type=int, default=64,
                          help='dataset tot use')

dataset_args.add_argument('--exclude-existing', default=False, action='store_true',
                          help='dataset tot use')

dataset_args.add_argument('--nprocs', type=int, default=50,
                          help='dataset tot use')

# dataset_args.add_argument('-c', '--category', type=str, default='tilde_cubes_3_20000',
#                           help='path to datasets')
# dataset_args.add_argument('--model-type', type=str, default='common', # 
#                           help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# dataset_args.add_argument('--apply-random-flipping', type=bool, default=False, # false flipping 
#                           help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# # 
# dataset_args.add_argument('--apply-random-shift', type=bool, default=False, # 
#                           help='type of model to use here [comon, part-tree, two-parts, three-parts]')


opt = parser.parse_args()
