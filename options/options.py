from . import HierarchyArgmentParser

parser = HierarchyArgmentParser()


# Experiment arguments
exp_args = parser.add_parser("experiment")
exp_args.add_argument('--experiment-id', type=str, default='pretraining',
                      help='experiment id')
exp_args.add_argument('--model-dir', type=str, default='./ckpts',
                      help='root path to svae the model')
exp_args.add_argument('--model-flag', type=str, default='single_mask', # more_mask
                      help='experiment flag')
exp_args.add_argument('-s', '--seed', type=int, default=2913,
                      help='random seed')
exp_args.add_argument('--run-mode', type=str, default='train',
                      help='train | eval | test')

# exp_args.add_argument('--sv-samples-dir', type=str, default='./samples/exp',
#                       help='train | eval | test')

# Experiment arguments
dataset_args = parser.add_parser("dataset")
dataset_args.add_argument('--dataset-name', type=str, default='syn_cubes',
                          help='dataset tot use')
dataset_args.add_argument('-c', '--category', type=str, default='tilde_cubes_3_20000',
                          help='path to datasets')
dataset_args.add_argument('--model-type', type=str, default='common', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--apply-random-flipping', type=bool, default=False, # false flipping 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# 
dataset_args.add_argument('--apply-random-shift', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# 
# dataset_args.add_argument('--apply-random-scaling', type=bool, default=False, # 
                          # help='type of model to use here [comon, part-tree, two-parts, three-parts]')
# 
dataset_args.add_argument('--apply-random-scaling', type=int, default=0, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')


dataset_args.add_argument('--st-subd-idx', type=int, default=0, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# 
dataset_args.add_argument('--apply-random-warping', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--shuffle-vertices', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--max-permit-vertices', type=int, default=800, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--max-permit-faces', type=int, default=2800, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--sv-samples-dir', type=str, default='./samples/exp',
                      help='train | eval | test')

dataset_args.add_argument('--lab', type=str, default='thu',
                      help='train | eval | test')

dataset_args.add_argument('--dataset-root-path', type=str, default='/nas/datasets/gen/datasets',
                      help='train | eval | test')
dataset_args.add_argument('--mask-low-ratio', type=float, default=0.15, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')
dataset_args.add_argument('--mask-high-ratio', type=float, default=0.16, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--mask-vertices-type', type=str, default='coord',
                      help='train | eval | test')

dataset_args.add_argument('--cus-perm-xl', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--category-part-indicator', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--coarse-angle-limit', type=int, default=30, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--fine-angle-limit', type=int, default=15, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--use-normalized-data', type=int, default=1, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--nn-vertices-predict-ratio', type=float, default=1.0, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--use-eyeglasses-frame', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--not-remove-du', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')
# dataset_args.add_argument('--cut-vertices', type=int, default=0, # 
                          # help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--sort-dist', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--data-type', type=str, default='binvox',
                      help='train | eval | test')

dataset_args.add_argument('--load-meta', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--use-inst', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--num-workers', type=int, default=10, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--num-parts', type=int, default=1, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--num-objects', type=int, default=1, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')


dataset_args.add_argument('--balance-classes', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')
# use_context_window_as_max
dataset_args.add_argument('--use-context-window-as-max', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

# use_context_window_as_max
dataset_args.add_argument('--ar-object', type=bool, default=False, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--half-part', default=False, action='store_true', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--use-part', default=False, action='store_true', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--subdn', type=int, default=2, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--subdn-test', type=int, default=2, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--ar-subd-idx', type=int, default=2, # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--fake-upsample', default=False, action='store_true', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--use-local-frame', default=False, action='store_true', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

dataset_args.add_argument('--min-quant-range', type=float, default=-0.5, help='learning rate')
dataset_args.add_argument('--max-quant-range', type=float, default=0.5, help='learning rate')

# Experiment arguments
loss_args = parser.add_parser("loss")
loss_args.add_argument('--learning-rate', type=float, default=5e-4, help='learning rate')

loss_args.add_argument('--training-steps', type=int, default=1000000, help='training steps')

loss_args.add_argument('--check-step', type=int, default=2000, help='maximum sample legnth')

loss_args.add_argument('--batch-size', type=int, default=1, help='batch size')

loss_args.add_argument('--with-laplacian', default=False, action='store_true', # 
                          help='type of model to use here [comon, part-tree, two-parts, three-parts]')

### options ###

# Sampling arguments
sampling_args = parser.add_parser("sampling")
sampling_args.add_argument('--feature-extraction', default=False, action='store_true', help='learning rate')
sampling_args.add_argument('--test-train', default=False, action='store_true', help='learning rate')

sampling_args.add_argument('--encode-and-sample', default=False, action='store_true', help='learning rate')

# sampling_args.add_argument('--training-steps', type=int, default=1000000, help='training steps')

# sampling_args.add_argument('--check-step', type=int, default=2000, help='maximum sample legnth')

# sampling_args.add_argument('--batch-size', type=int, default=1, help='batch size')

### options ###


common_args = parser.add_parser("common")

common_args.add_argument('--exp-flag', type=str, default="", help='batch size')
common_args.add_argument('--exp-mode', type=str, default="pretraining", help='batch size')

# # common arguments for both the vertex model and the face model
model_args = parser.add_parser("model")

model_args.add_argument('--part-tree', type=dict, default=None, help='part tree for the part structure')

model_args.add_argument('--use-light-weight', type=bool, default=False, help='whether to use light weight model')

model_args.add_argument('--quantization-bits', type=int, default=8, help='numberf of bits for vertex model cooridnate quantization')

model_args.add_argument('--use-discrete-embeddings', type=bool, default=True, help='whether to use discrete embeddings')

### True for single objects, 
model_args.add_argument('--recenter-mesh', type=bool, default=True, help='maximum sample legnth')

model_args.add_argument('--only-return-complete', type=bool, default=False, help='maximum sample legnth')

model_args.add_argument('--num-classes', type=int, default=1, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--post-process', type=bool, default=False, help='maximum sample legnth')

model_args.add_argument('--top-p', type=float, default=0.90, help='maximum sample legnth')

model_args.add_argument('--memory-efficient', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--sv-model-dir', type=str, default='ckpts/common', help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--sv-model-freq', type=int, default=20000, help='maximum sample legnth')

model_args.add_argument('--vertex-model-path', type=str, default="", help='maximum sample legnth')

model_args.add_argument('--face-model-path', type=str, default="", help='maximum sample legnth')

model_args.add_argument('--fine-vertex-model-path', type=str, default="", help='maximum sample legnth')

model_args.add_argument('--fine-face-model-path', type=str, default="", help='maximum sample legnth')

model_args.add_argument('--pc-encoder-model-path', type=str, default="", help='maximum sample legnth')


model_args.add_argument('--finetune', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--debug', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--cross', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--sample-class-idx', type=int, default=0, help='number of classes for training, if we use class conditioning')


model_args.add_argument('--prefix-key-len', type=int, default=1, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--prefix-value-len', type=int, default=1, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--specify-light-weight', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--beam-search', type=bool, default=False, help='memory efficient mechanism in decoder and encoder')

model_args.add_argument('--encoder-adapter-path', type=str, default="", help='maximum sample legnth')
model_args.add_argument('--decoder-adapter-face-path', type=str, default="", help='maximum sample legnth')
model_args.add_argument('--decoder-adapter-vertex-path', type=str, default="", help='maximum sample legnth')

model_args.add_argument('--context-window', type=int, default=-1, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--cut-vertices', type=int, default=0, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--nn-training-categories', type=int, default=9, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--shift-context-window', type=bool, default=False, help='number of classes for training, if we use class conditioning')

model_args.add_argument('--with-prompt-encoder', type=bool, default=False, help='number of classes for training, if we use class conditioning')

# sampling_max_num_grids
model_args.add_argument('--sampling-max-num-grids', type=int, default=-1, help='number of classes for training, if we use class conditioning')

## adaptation_optimization
model_args.add_argument('--adaptation-optimization', default=False, action='store_true', help='number of classes for training, if we use class conditioning')


model_args.add_argument('--use-prob', default=False, action='store_true', help='number of classes for training, if we use class conditioning')

model_args.add_argument('--use-half-flaps', default=False, action='store_true', help='number of classes for training, if we use class conditioning')

model_args.add_argument('--pred-delta', default=False, action='store_true', help='number of classes for training, if we use class conditioning')

model_args.add_argument('--pred-prob', default=False, action='store_true', help='number of classes for training, if we use class conditioning')

model_args.add_argument('--prompt-tuning', default=False, action='store_true', help='number of classes for training, if we use class conditioning')
# # vertex model arguments
vertex_model_args = parser.add_parser("vertex_model")

# vertex_model_args.add_argument('--num-classes', type=int, default=1, help='number of classes for training, if we use class conditioning')

# vertex_model_args.add_argument('--quantization-bits', type=int, default=8, help='numberf of bits for vertex model cooridnate quantization')

vertex_model_args.add_argument('--vertex-class-conditional', type=bool, default=True, help='whether to use class conditional for multi-class setting')

vertex_model_args.add_argument('--max-vertices', type=int, default=850, help='maximum number of vertices allowed for input')

vertex_model_args.add_argument('--inter-part-auto-regressive', type=bool, default=True, help='whether to use inter-part auto-regressive')

vertex_model_args.add_argument('--predict-joint', type=bool, default=True, help='whether to predict joints')

# vertex_model_args.add_argument('--use-discrete-embeddings', type=bool, default=True, help='whether to use discrete embeddings')

vertex_model_args.add_argument('--max-sample-length', type=int, default=800, help='maximum sample legnth')

vertex_model_args.add_argument('--max-num-grids', type=int, default=400, help='maximum sample legnth')

vertex_model_args.add_argument('--grid-size', type=int, default=4, help='maximum sample legnth')

# vertex_model_args.add_argument('--top-p', type=float, default=0.90, help='maximum sample legnth')

# vertex_model_args.add_argument('--recenter-mesh', type=bool, default=True, help='maximum sample legnth')

# vertex_model_args.add_argument('--recenter-mesh', type=bool, default=True, help='maximum sample legnth')




# # face model arguments
face_model_args = parser.add_parser("face_model")

face_model_args.add_argument('--face-class-conditional', type=bool, default=False, help='whether to use inter-part auto-regressive')

face_model_args.add_argument('--max-faces', type=int, default=8412, help='maximum number of face sequence allowed for input')

# face_model_args.add_argument('--quantization-bits', type=int, default=8, help='numberf of bits for vertex model cooridnate quantization')

face_model_args.add_argument('--decoder-cross-attention', type=bool, default=True, help='whether to use cross attention in the decoder')

face_model_args.add_argument('--max-sample-length', type=int, default=8000, help='maximum number of face sequence allowed for input')



# face_model_args.add_argument('--top-p', type=float, default=0.90, help='maximum sample legnth')

# face_model_args.add_argument('--use-discrete-vertex-embeddings', type=bool, default=True, help='whether to use vertex discrete embeddings')

opt = parser.parse_args()
