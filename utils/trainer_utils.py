from options.options import opt

# import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch
import os



def reduce_mean(tensor, nprocs):
    rt = tensor # .clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / nprocs
    return rt


def ddp_init(): # initialize data parallel settings
    #### init_process_group ####
    torch.distributed.init_process_group(backend='nccl')
    tmpp_local_rnk = int(os.environ['LOCAL_RANK'])
    print("os_environed:", tmpp_local_rnk)
    local_rank = tmpp_local_rnk
    torch.cuda.set_device(local_rank)
    opt.device = local_rank

    ''' Set number of procs '''
    opt.nprocs = torch.cuda.device_count()
    print("device count:", opt.nprocs)
    nprocs = opt.nprocs
    nprocs = nprocs

    _use_multi_gpu = True

    cudnn.benchmark = True
    return local_rank, _use_multi_gpu


def get_model_configs(use_light_weight=None):
    # use_light_weight_model = True
    if use_light_weight is None:
        use_light_weight_model = opt.model.use_light_weight
    else:
        use_light_weight_model =use_light_weight
    # use_light_weight_model = False
    # memory_efficient = True
    # part_tree = {'idx': 0}
  
    vertex_module_config = dict(
        encoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=10,
            dropout_rate=0.2,
            re_zero=True, # 
        ),
        decoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=24,
            dropout_rate=0.4,
            re_zero=True,
        ),
        num_classes=opt.model.num_classes,
        quantization_bits=opt.model.quantization_bits,
        class_conditional=opt.vertex_model.vertex_class_conditional,
        max_num_input_verts=opt.vertex_model.max_vertices,
        # inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
        # predict_joint=opt.vertex_model.predict_joint,
        use_discrete_embeddings=opt.model.use_discrete_embeddings,
        # part_tree=part_tree
    )
    #### vertex module config ####

    #### vertex module config light ####
    vertex_module_config_light = dict(
        encoder_config=dict(
        hidden_size=128 if use_light_weight is None else 512,
        fc_size=512,
        num_heads=4,
        layer_norm=True,
        num_layers=3,
        dropout_rate=0.2,
        re_zero=True,
        ),
        decoder_config=dict(
        hidden_size=128 if use_light_weight is None else 512,
        fc_size=512,
        num_heads=4,
        layer_norm=True,
        num_layers=3,
        dropout_rate=0.4,
        re_zero=True,
        ),
        num_classes=opt.model.num_classes,
        quantization_bits=opt.model.quantization_bits,
        class_conditional=opt.vertex_model.vertex_class_conditional,
        max_num_input_verts=opt.vertex_model.max_vertices,
        # inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
        # predict_joint=opt.vertex_model.predict_joint,
        use_discrete_embeddings=opt.model.use_discrete_embeddings,
        # part_tree=part_tree
    )
    #### vertex module config light ####

    #### face module config ####
    face_module_config=dict(
        encoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=10,
            dropout_rate=0.2,
            re_zero=True,
        ),
        decoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=14,
            dropout_rate=0.2,
            re_zero=True,
        ),
        class_conditional=opt.face_model.face_class_conditional,
        quantization_bits=opt.model.quantization_bits,
        decoder_cross_attention=opt.face_model.decoder_cross_attention,
        use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
        max_seq_length=opt.face_model.max_faces,
        # part_tree={'idx': 0}
    )
    #### face module config ####

    #### face module config light ####
    face_module_config_light = dict(
        encoder_config=dict(
            hidden_size=128 if use_light_weight is None else 512,
            fc_size=512,
            num_heads=4,
            layer_norm=True,
            num_layers=3,
            dropout_rate=0.2,
            re_zero=True,
        ),
        decoder_config=dict(
            hidden_size=128 if use_light_weight is None else 512,
            fc_size=512,
            num_heads=4,
            layer_norm=True,
            num_layers=3,
            dropout_rate=0.2,
            re_zero=True,
        ),
        class_conditional=opt.face_model.face_class_conditional,
        quantization_bits=opt.model.quantization_bits,
        decoder_cross_attention=opt.face_model.decoder_cross_attention,
        use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
        max_seq_length=opt.face_model.max_faces,
        # part_tree={'idx': 0}
    )
    vertex_module_config = vertex_module_config_light if use_light_weight_model else vertex_module_config
    face_module_config = face_module_config_light if use_light_weight_model else face_module_config
    return vertex_module_config, face_module_config


def get_model_configs_v2(use_light_weight=None):
    # use_light_weight_model = True
    if use_light_weight is None:
        use_light_weight_model = opt.model.use_light_weight
    else:
        use_light_weight_model =use_light_weight
    # use_light_weight_model = False
    # memory_efficient = True
    # part_tree = {'idx': 0}
  
    vertex_module_config = dict(
        encoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=14,
            dropout_rate=0.2,
            re_zero=True, # 
        ),
        decoder_config=dict(
            hidden_size=1024,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=36,
            dropout_rate=0.4,
            re_zero=True,
        ),
        num_classes=opt.model.num_classes,
        quantization_bits=opt.model.quantization_bits,
        class_conditional=opt.vertex_model.vertex_class_conditional,
        max_num_input_verts=opt.vertex_model.max_vertices,
        # inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
        # predict_joint=opt.vertex_model.predict_joint,
        use_discrete_embeddings=opt.model.use_discrete_embeddings,
        # part_tree=part_tree
    )
    #### vertex module config ####

    #### face module config ####
    face_module_config=dict(
        encoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=14,
            dropout_rate=0.2,
            re_zero=True,
        ),
        decoder_config=dict(
            hidden_size=512,
            fc_size=2048,
            num_heads=8,
            layer_norm=True,
            num_layers=14,
            dropout_rate=0.2,
            re_zero=True,
        ),
        class_conditional=opt.face_model.face_class_conditional,
        quantization_bits=opt.model.quantization_bits,
        decoder_cross_attention=opt.face_model.decoder_cross_attention,
        use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
        max_seq_length=opt.face_model.max_faces,
        # part_tree={'idx': 0}
    )
    #### face module config ####

    vertex_module_config =  vertex_module_config
    face_module_config = face_module_config
    return vertex_module_config, face_module_config


def get_adapter_config():
    # hidden_size=512,
            # fc_size=2048,
            # num_heads=8,
            # layer_norm=True,
            # num_layers=14,
            # dropout_rate=0.2,
            # re_zero=True,
    use_light_weight_model = opt.model.use_light_weight
    adapter_module_config_encoder = dict(
        hidden_size=512,
        down_r_size=256,
        num_layers=10
    )
    #### vertex module config ####


    adapter_module_config_decoder_face = dict(
        hidden_size=512,
        down_r_size=256,
        num_layers=14
    )

    adapter_module_config_decoder_vertex = dict(
        hidden_size=512,
        down_r_size=256,
        num_layers=24
    )

    adapter_module_config_encoder_light = dict(
        hidden_size=128,
        down_r_size=64,
        num_layers=3
    )

    adapter_module_config_decoder_face_light = dict(
        hidden_size=128,
        down_r_size=64,
        num_layers=3
    )

    adapter_module_config_decoder_vertex_light = dict(
        hidden_size=128,
        down_r_size=64,
        num_layers=3
    )

    if use_light_weight_model:
        return adapter_module_config_encoder_light, adapter_module_config_decoder_vertex_light, adapter_module_config_decoder_face_light
    else:
        return adapter_module_config_encoder, adapter_module_config_decoder_vertex, adapter_module_config_decoder_face


    # adapter_module_config = adapter_module_config_light if use_light_weight_model else adapter_module_config
    # return adapter_module_config


def get_prefix_key_value_config():
    use_light_weight_model = opt.model.use_light_weight
    prefix_key_val_len = 64
    prefix_key_len = opt.model.prefix_key_len
    prefix_value_len = opt.model.prefix_value_len
    adapter_module_config = dict(
        hidden_size=512,
        prefix_key_len=prefix_key_len,
        prefix_value_len=prefix_value_len,
        num_layers=24
    )
    #### vertex module config ####

    adapter_module_config_light = dict(
        hidden_size=128,
        prefix_key_len=16,
        prefix_value_len=16,
        num_layers=3
    )

    adapter_module_config = adapter_module_config_light if use_light_weight_model else adapter_module_config
    return adapter_module_config

def get_prompt_value_config():
    use_light_weight_model = opt.model.use_light_weight
    prefix_key_val_len = 64
    prefix_key_len = opt.model.prefix_key_len
    prefix_value_len = opt.model.prefix_value_len
    adapter_module_config = dict(
        hidden_size=512,
        prefix_key_len=prefix_key_len,
        prefix_value_len=prefix_value_len,
        num_layers=24,
        nn_prompts=3
    )
    #### vertex module config ####

    adapter_module_config_light = dict(
        hidden_size=128,
        prefix_key_len=16,
        prefix_value_len=16,
        num_layers=3,
        nn_prompts=3
    )

    adapter_module_config = adapter_module_config_light if use_light_weight_model else adapter_module_config
    return adapter_module_config

def get_prompt_prompt_value_config():
    # hidden_size=256,
    #              fc_size=512,
    #              prefix_key_len=16,
    #              prefix_value_len=16,
    #              num_layers=3, # we use three attention layers?
    #              nn_prompts=3,
    #              num_heads=4,
    #              layer_norm=True,
    #              re_zero=True,
    #              dropout_rate=0.4)
    prefix_key_len = opt.model.prefix_key_len
    prefix_value_len = opt.model.prefix_value_len
    adapter_module_config = dict(
        hidden_size=512,
        fc_size=512,
        prefix_key_len=prefix_key_len,
        prefix_value_len=prefix_value_len,
        num_layers=3,
        nn_prompts=3,
        num_heads=4,
        layer_norm=True,
        re_zero=True,
        dropout_rate=0.4
    )
    return adapter_module_config

###### From trainer/training_1.py ######
# def get_model_configs():
#     # use_light_weight_model = True
#     use_light_weight_model = opt.model.use_light_weight
#     # use_light_weight_model = False
#     memory_efficient = True
#     part_tree = {'idx': 0}
  
#     vertex_module_config = dict(
#         encoder_config=dict(
#         hidden_size=512,
#         fc_size=2048,
#         num_heads=8,
#         layer_norm=True,
#         num_layers=10,
#         dropout_rate=0.2,
#         re_zero=True, # 
#         ),
#         decoder_config=dict(
#         hidden_size=512,
#         fc_size=2048,
#         num_heads=8,
#         layer_norm=True,
#         num_layers=24,
#         dropout_rate=0.4,
#         re_zero=True,
#         ),
#         num_classes=opt.model.num_classes,
#         quantization_bits=opt.model.quantization_bits,
#         class_conditional=opt.vertex_model.vertex_class_conditional,
#         max_num_input_verts=opt.vertex_model.max_vertices,
#         inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
#         predict_joint=opt.vertex_model.predict_joint,
#         use_discrete_embeddings=opt.model.use_discrete_embeddings,
#         # part_tree=part_tree
#     )
#     #### vertex module config ####

#     #### vertex module config light ####
#     vertex_module_config_light = dict(
#         encoder_config=dict(
#         hidden_size=128,
#         fc_size=512,
#         num_heads=4,
#         layer_norm=True,
#         num_layers=3,
#         dropout_rate=0.2,
#         re_zero=True,
#         ),
#         decoder_config=dict(
#         hidden_size=128,
#         fc_size=512,
#         num_heads=4,
#         layer_norm=True,
#         num_layers=3,
#         dropout_rate=0.4,
#         re_zero=True,
#         ),
#         num_classes=opt.model.num_classes,
#         quantization_bits=opt.model.quantization_bits,
#         class_conditional=opt.vertex_model.vertex_class_conditional,
#         max_num_input_verts=opt.vertex_model.max_vertices,
#         inter_part_auto_regressive=opt.vertex_model.inter_part_auto_regressive,
#         predict_joint=opt.vertex_model.predict_joint,
#         use_discrete_embeddings=opt.model.use_discrete_embeddings,
#         # part_tree=part_tree
#     )
#     #### vertex module config light ####

#     #### face module config ####
#     face_module_config=dict(
#         encoder_config=dict(
#             hidden_size=512,
#             fc_size=2048,
#             num_heads=8,
#             layer_norm=True,
#             num_layers=10,
#             dropout_rate=0.2,
#             re_zero=True,
#         ),
#         decoder_config=dict(
#             hidden_size=512,
#             fc_size=2048,
#             num_heads=8,
#             layer_norm=True,
#             num_layers=14,
#             dropout_rate=0.2,
#             re_zero=True,
#         ),
#         class_conditional=opt.face_model.face_class_conditional,
#         quantization_bits=opt.model.quantization_bits,
#         decoder_cross_attention=opt.face_model.decoder_cross_attention,
#         use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
#         max_seq_length=opt.face_model.max_faces,
#         # part_tree={'idx': 0}
#     )
#     #### face module config ####

#     #### face module config light ####
#     face_module_config_light = dict(
#         encoder_config=dict(
#             hidden_size=128,
#             fc_size=512,
#             num_heads=4,
#             layer_norm=True,
#             num_layers=3,
#             dropout_rate=0.2,
#             re_zero=True,
#         ),
#         decoder_config=dict(
#             hidden_size=128,
#             fc_size=512,
#             num_heads=4,
#             layer_norm=True,
#             num_layers=3,
#             dropout_rate=0.2,
#             re_zero=True,
#         ),
#         class_conditional=opt.face_model.face_class_conditional,
#         quantization_bits=opt.model.quantization_bits,
#         decoder_cross_attention=opt.face_model.decoder_cross_attention,
#         use_discrete_vertex_embeddings=opt.model.use_discrete_embeddings,
#         max_seq_length=opt.face_model.max_faces,
#         # part_tree={'idx': 0}
#     )
#     vertex_module_config = vertex_module_config_light if use_light_weight_model else vertex_module_config
#     face_module_config = face_module_config_light if use_light_weight_model else face_module_config
#     return vertex_module_config, face_module_config
###### From trainer/training_1.py ######
