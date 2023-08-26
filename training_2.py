# !pip install tensorflow==1.15 dm-sonnet==1.36 tensor2tensor==1.14

import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt

# %cd /tmp
# %rm -rf /tmp/deepmind_research
# !git clone https://github.com/deepmind/deepmind-research.git \
#   /tmp/deepmind_research
# %cd /tmp/deepmind_research/polygen
import modules
import data_utils


# Prepare synthetic dataset
##### Process toy dataset #####
# ex_list = []
# for k, mesh in enumerate(['cube', 'cylinder', 'cone', 'icosphere']):
#   mesh_dict = data_utils.load_process_mesh(
#       os.path.join('meshes', '{}.obj'.format(mesh)))
#   mesh_dict['class_label'] = k
#   ex_list.append(mesh_dict)
# synthetic_dataset = tf.data.Dataset.from_generator(
#     lambda: ex_list,
#     output_types={
#         'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32},
#     output_shapes={
#         'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]),
#         'class_label': tf.TensorShape(())}
#     )
# ex = synthetic_dataset.make_one_shot_iterator().get_next()
# nn_cat = 4
##### Process toy dataset #####

cat_folder_names = ["two_pieces_mesh_part_1", "two_pieces_mesh_part_2"]
cat_synthetic_datasets = []
cat_exs = []

tot_part_ex_list = []
for i_part, part_folder_nm in enumerate(cat_folder_names):
  part_instance_mesh_fns = os.listdir(part_folder_nm)
  part_ex_list = []
  for k, part_mesh_fn in enumerate(part_instance_mesh_fns):
    if part_mesh_fn.endswith(".obj"):
      mesh_dict = data_utils.load_process_mesh( # load processed mesh? --- for the mesh dict
      os.path.join(part_folder_nm, part_mesh_fn)) # mesh_fn --- cat folder name
      mesh_dict['class_label'] = 0
      part_ex_list.append(mesh_dict) # add the mesh to the ex_list
    else:
      continue
  tot_part_ex_list.append(part_ex_list)

  # # ex_list = ex_list[:10]
  # part_synthetic_dataset = tf.data.Dataset.from_generator(
  #   lambda: part_ex_list, # output_types
  #   output_types={
  #       'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32}, # synthetic dataset
  #   output_shapes={ # faces
  #       'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]), # none for faces?
  #       'class_label': tf.TensorShape(())} # output shape --- vertices; # output shape, vertices, faces, and class_label
  # )
  # part_ex = part_synthetic_dataset.make_one_shot_iterator().get_next()
  # # s

  # cat_synthetic_datasets.append(part_synthetic_dataset)
  # cat_exs.append(part_ex)
for i_part in range(len(cat_folder_names)):
  # add synthetic dataset and syn_dataset_ex here
  cat_synthetic_datasets.append( tf.data.Dataset.from_generator(
    lambda: tot_part_ex_list[i_part], # output_types
    output_types={
        'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32}, # synthetic dataset
    output_shapes={ # faces
        'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]), # none for faces?
        'class_label': tf.TensorShape(())} # output shape --- vertices; # output shape, vertices, faces, and class_label
  ))
  cat_exs.append(cat_synthetic_datasets[-1].make_one_shot_iterator().get_next())



##### Process toy dataset for laptops #####
# ex_list = []
# # cat_folder_name = "laptops_objs"
# # cat_folder_name = "laptop_simplified_v4"
# # cat_folder_name = "laptop_sim_v5"
# cat_folder_name = "two_pieces_mesh"
# cat_instance_mesh_fns = os.listdir(cat_folder_name) # instances
# for k, mesh_fn in enumerate(cat_instance_mesh_fns):
#   # if mesh_fn.endswith(".obj") and k < 11:
#   # mesh_fn.endswith(".obj")
#   if mesh_fn.endswith(".obj"):
#     mesh_dict = data_utils.load_process_mesh( # load processed mesh? --- for the mesh dict
#       os.path.join(cat_folder_name, mesh_fn)) # mesh_fn --- cat folder name
#     mesh_dict['class_label'] = 0
#     ex_list.append(mesh_dict) # add the mesh to the ex_list
#   else:
#     continue

# # ex_list = ex_list[:10]
# synthetic_dataset = tf.data.Dataset.from_generator(
#     lambda: ex_list,
#     output_types={
#         'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32}, # synthetic dataset
#     output_shapes={ # faces
#         'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]), # none for faces?
#         'class_label': tf.TensorShape(())} # output shape --- vertices; # output shape, vertices, faces, and class_label
#     )
# ex = synthetic_dataset.make_one_shot_iterator().get_next()


nn_cat = 1 # number of categories

# synthetic dataset --- Dataset from generator
#


# Inspect the first mesh
# with tf.Session() as sess:
#   ex_np = sess.run(ex)
# print(ex_np)


with tf.Session() as sess:
  for i_part, part_ex in enumerate(cat_exs):
    part_ex_np = sess.run(part_ex) # part_ex
  # ex_np = sess.run(ex)
    print(part_ex_np) # part_ex_np
    mesh_list = []
    # for ii in range(4):
    #   cur_part_ex = sess.run(part_ex_np[ii]) # p
    #   mesh_list.append(
    #     {
    #       'vertices': data_utils.dequantize_verts(cur_part_ex['vertices']),
    #       'faces': data_utils.unflatten_faces(cur_part_ex['faces'])
    #     }
    #   )
    mesh_list.append(
        {
          'vertices': data_utils.dequantize_verts(part_ex_np['vertices']),
          'faces': data_utils.unflatten_faces(part_ex_np['faces'])
        }
      )
    data_utils.plot_meshes(mesh_list[:4], ax_lims=0.4)

# Plot the meshes
# mesh_list = []
# with tf.Session() as sess: # mehs list
#   for i in range(nn_cat):
#     ex_np = sess.run(cat_exs[i_part])
#     mesh_list.append(
#         {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
#         'faces': data_utils.unflatten_faces(ex_np['faces'])})
#   data_utils.plot_meshes(mesh_list[:4], ax_lims=0.4)





# Plot the meshes
# mesh_list = []
# with tf.Session() as sess:
#   for i in range(4):
#     ex_np = sess.run(ex)
#     mesh_list.append(
#         {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
#          'faces': data_utils.unflatten_faces(ex_np['faces'])})
# data_utils.plot_meshes(mesh_list[:4], ax_lims=0.4)

# maxx_nn_vertices = 0
# maxx_nn_faces = 0
# with tf.Session() as sess:
#   for i in range(100):
#     ex_np = sess.run(ex)
#     nn_vertices = ex_np['vertices'].shape[0]
#     nn_faces = ex_np['faces'].shape[0]
#     maxx_nn_vertices = max(maxx_nn_vertices, nn_vertices)
#     maxx_nn_faces = max(nn_faces, maxx_nn_faces)
#     print(i, ", nn_vertices:", nn_vertices, ", nn_faces:", nn_faces)
#     # mesh_list.append(
#     #     {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
#     #      'faces': data_utils.unflatten_faces(ex_np['faces'])})
# print("maxx_nn_vertices:", maxx_nn_vertices, "maxx_nn_faces:", maxx_nn_faces)



# maxx_nn_vertices = 0
# maxx_nn_faces = 0

print("length of cat_exs: ", len(cat_exs))
with tf.Session() as sess:
  for i_part, part_ex in enumerate(cat_exs):
    maxx_nn_vertices = 0
    maxx_nn_faces = 0
    for i in range(100):
      ex_np = sess.run(part_ex)
      nn_vertices = ex_np['vertices'].shape[0]
      nn_faces = ex_np['faces'].shape[0]
      maxx_nn_vertices = max(maxx_nn_vertices, nn_vertices)
      maxx_nn_faces = max(nn_faces, maxx_nn_faces)
      print(i, ", nn_vertices:", nn_vertices, ", nn_faces:", nn_faces)
      # mesh_list.append(
      #     {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
      #      'faces': data_utils.unflatten_faces(ex_np['faces'])})
    print("maxx_nn_vertices:", maxx_nn_vertices, "maxx_nn_faces:", maxx_nn_faces)







nn_cat = 1

vertex_model_datasets = []
vertex_model_batches = []
# cat_synthetic_datasets
# for cur_part_syn_dataset in cat_synthetic_datasets:
#     cur_part_vertex_model_dataset = data_utils.make_vertex_model_dataset(
#     cur_part_syn_dataset, apply_random_shift=False) # vertex model dataset
#     cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.repeat()
#     # vertex model dataset
#     cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.padded_batch(
#     4, padded_shapes=cur_part_vertex_model_dataset.output_shapes)
#     cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.prefetch(1)
#     cur_part_vertex_model_batch = cur_part_vertex_model_dataset.make_one_shot_iterator().get_next() # get context? get vertex model
#     vertex_model_datasets.append(cur_part_vertex_model_dataset)
#     vertex_model_batches.append(cur_part_vertex_model_batch)

for i_p in range(len(cat_synthetic_datasets)):
    # cur_part_vertex_model_dataset = data_utils.make_vertex_model_dataset(
    # cat_synthetic_datasets[i_p], apply_random_shift=False) # vertex model dataset
    vertex_model_datasets.append(data_utils.make_vertex_model_dataset(
        cat_synthetic_datasets[i_p], apply_random_shift=False))

for i_p in range(len(cat_synthetic_datasets)):
    vertex_model_datasets[i_p] = vertex_model_datasets[i_p].repeat()
    vertex_model_datasets[i_p] = vertex_model_datasets[i_p].padded_batch(
    4, padded_shapes=vertex_model_datasets[i_p].output_shapes)
    vertex_model_datasets[i_p] = vertex_model_datasets[i_p].prefetch(1)
    # vertex model batches
    vertex_model_batches.append(vertex_model_datasets[i_p].make_one_shot_iterator().get_next())

    # cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.repeat()
    # # vertex model dataset
    # cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.padded_batch(
    # 4, padded_shapes=cur_part_vertex_model_dataset.output_shapes)

    # cur_part_vertex_model_dataset = cur_part_vertex_model_dataset.prefetch(1)
    # cur_part_vertex_model_batch = cur_part_vertex_model_dataset.make_one_shot_iterator().get_next() # get context? get vertex model
    # vertex_model_datasets.append(cur_part_vertex_model_dataset)
    # vertex_model_batches.append(cur_part_vertex_model_batch)


# , decoder_config, class_conditional=False, num_classes=55, name='part_tree'
simple_part_tree = modules.SimplePartTreeGenModule(
    decoder_config={
        'hidden_size': 128,
        'fc_size': 512,
        'num_layers': 3,
        'dropout_rate': 0.
    },
    class_conditional=True,
    num_classes=nn_cat,
)

# need two vertex model dataset
# vertex_model_pred_dist = vertex_model(vertex_model_batch)
# part_node
part_node_pred_dict = simple_part_tree(vertex_model_batches[0])
# rt_dict = {"left": left_node_feature, "rgt": rgt_node_feature, "joint_axis": joint_axis_dir, "joint_pvp": joint_pvp}
left_part_feature, rgt_part_feature, joint_axis, joint_pvv = part_node_pred_dict['left'], part_node_pred_dict['rgt'], part_node_pred_dict['joint_axis'], part_node_pred_dict['joint_pvp']

vertex_model_batches[0]['global_context'] = left_part_feature
vertex_model_batches[1]['global_context'] = rgt_part_feature


# # Prepare the dataset for vertex model training
# vertex_model_dataset = data_utils.make_vertex_model_dataset(
#     synthetic_dataset, apply_random_shift=False) # vertex model dataset
# vertex_model_dataset = vertex_model_dataset.repeat()
# # vertex model dataset
# vertex_model_dataset = vertex_model_dataset.padded_batch(
#     4, padded_shapes=vertex_model_dataset.output_shapes)
# vertex_model_dataset = vertex_model_dataset.prefetch(1)
# vertex_model_batch = vertex_model_dataset.make_one_shot_iterator().get_next() # get context? get vertex model


# nn_max_vertices = 250
# nn_max_vertices = 11841
nn_max_vertices = 1300 # 2000
# nn_max_vertices = 3500
# nn_max_faces = 500
# nn_max_faces = 150000
# nn_max_faces = 3264
# nn_max_faces = 11841
nn_max_faces = 4300
# prev 9788

# Create vertex model
# vertex_model = modules.VertexModel(
#     decoder_config={
#         'hidden_size': 128,
#         'fc_size': 512,
#         'num_layers': 3,
#         'dropout_rate': 0.
#     },
#     class_conditional=True,
#     num_classes=nn_cat,
#     max_num_input_verts=nn_max_vertices,
#     quantization_bits=8,
# )
# vertex_model_pred_dist = vertex_model(vertex_model_batch)
# # vertex model loss --- vertices prediction loss
# vertex_model_loss = -tf.reduce_sum(
#     vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) *
#     vertex_model_batch['vertices_flat_mask'])
# vertex_samples = vertex_model.sample( # vertex_model.sample
#     4, context=vertex_model_batch, max_sample_length=nn_max_vertices, top_p=0.95, # top_p
#     recenter_verts=False, only_return_complete=False)

# print(vertex_model_batch)
# print(vertex_model_pred_dist)
# print(vertex_samples)


### create vertex models ###
vertex_models = []
vertex_samples = []
vertex_model_losses = []
vertex_model_pred_dists = []
for i_v, cur_part_vertex_model_batch in enumerate(vertex_model_batches):
    cur_part_vertex_model = modules.VertexModel(
        decoder_config={
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        class_conditional=True,
        num_classes=nn_cat,
        max_num_input_verts=nn_max_vertices,
        quantization_bits=8,
        name=f"vertex_model_{i_v}"
    )
    # cur_part_vertex_model_pred_dist = cur_part_vertex_model(vertex_model_batches[i_v])
    vertex_model_pred_dists.append(cur_part_vertex_model(vertex_model_batches[i_v]))
    # vertex model loss --- vertices prediction loss
    # cur_part_vertex_model_loss = -tf.reduce_sum(
    #     vertex_model_pred_dists[-1].log_prob(vertex_model_batches[i_v]['vertices_flat']) *
    #     vertex_model_batches[i_v]['vertices_flat_mask'])
    # cur_part_vertex_samples = cur_part_vertex_model.sample( # vertex_model.sample
    #     4, context=vertex_model_batches[i_v], max_sample_length=nn_max_vertices, top_p=0.95, # top_p
    #     recenter_verts=False, only_return_complete=False)
    print(vertex_model_batches[i_v])
    print(vertex_model_pred_dists[-1])
    # print(cur_part_vertex_samples)
    vertex_models.append(cur_part_vertex_model)

    # vertex_samples.append(cur_part_vertex_samples)
    vertex_samples.append(cur_part_vertex_model.sample( # vertex_model.sample
        4, context=vertex_model_batches[i_v], max_sample_length=nn_max_vertices, top_p=0.95, # top_p
        recenter_verts=False, only_return_complete=False))

    # vertex_model_losses.append(cur_part_vertex_model_loss)
    vertex_model_losses.append(
        -tf.reduce_sum(
        vertex_model_pred_dists[-1].log_prob(vertex_model_batches[i_v]['vertices_flat']) *
        vertex_model_batches[i_v]['vertices_flat_mask'])
    )
### create vertex models ###


# face_model_dataset = data_utils.make_face_model_dataset(
#     synthetic_dataset, apply_random_shift=False)
# face_model_dataset = face_model_dataset.repeat()
# face_model_dataset = face_model_dataset.padded_batch(
#     4, padded_shapes=face_model_dataset.output_shapes)
# face_model_dataset = face_model_dataset.prefetch(1)
# face_model_batch = face_model_dataset.make_one_shot_iterator().get_next()

face_model_datasets = []
face_model_batches = []
for i_p, cur_part_syn_dataset in enumerate(cat_synthetic_datasets):
    face_model_datasets.append(data_utils.make_face_model_dataset(
        cat_synthetic_datasets[i_p], apply_random_shift=False))

for i_p in range(len(face_model_datasets)):
    # cur_part_face_model_dataset = data_utils.make_face_model_dataset(
    #     cat_synthetic_datasets[i_p], apply_random_shift=False)
    face_model_datasets[i_p] = face_model_datasets[i_p].repeat()
    face_model_datasets[i_p] = face_model_datasets[i_p].padded_batch(
        4, padded_shapes=face_model_datasets[i_p].output_shapes)
    face_model_datasets[i_p] = face_model_datasets[i_p].prefetch(1)
    face_model_batches.append(face_model_datasets[i_p].make_one_shot_iterator().get_next())
    # cur_part_face_model_dataset = cur_part_face_model_dataset.repeat()
    # cur_part_face_model_dataset = cur_part_face_model_dataset.padded_batch(
    #     4, padded_shapes=cur_part_face_model_dataset.output_shapes)
    # cur_part_face_model_dataset = cur_part_face_model_dataset.prefetch(1)
    # #
    # cur_part_face_model_batch = cur_part_face_model_dataset.make_one_shot_iterator().get_next()

    # face_model_datasets.append(cur_part_face_model_dataset)
    # face_model_batches.append(cur_part_face_model_batch)

# # Create face model
# # create face model for the model
# face_model = modules.FaceModel(
#     encoder_config={
#         'hidden_size': 128,
#         'fc_size': 512,
#         'num_layers': 3,
#         'dropout_rate': 0.
#     },
#     decoder_config={
#         'hidden_size': 128,
#         'fc_size': 512,
#         'num_layers': 3,
#         'dropout_rate': 0.
#     },
#     class_conditional=False,
#     max_seq_length=nn_max_faces,
#     quantization_bits=8,
#     decoder_cross_attention=True,
#     use_discrete_vertex_embeddings=True,
# )
# face_model_pred_dist = face_model(face_model_batch) # face_model for ...
# # face_model_loss ---- how to get face model loss? --- faces and meshes; losses
# face_model_loss = -tf.reduce_sum(
#     face_model_pred_dist.log_prob(face_model_batch['faces']) *
#     face_model_batch['faces_mask'])
# face_samples = face_model.sample(
#     context=vertex_samples, max_sample_length=nn_max_faces, top_p=0.95,
#     only_return_complete=False)
# print(face_model_batch)
# print(face_model_pred_dist)
# print(face_samples)

face_samples = []
face_models = []
face_model_losses = []
face_model_pred_dists = []
for i_p, cur_part_face_model_batch in enumerate(face_model_batches):
    cur_part_face_model = modules.FaceModel(
        encoder_config={  # need encoder for vertex encoding
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        decoder_config={  # decoder for faces decoding
            'hidden_size': 128,
            'fc_size': 512,
            'num_layers': 3,
            'dropout_rate': 0.
        },
        class_conditional=False,  # no class conditional
        max_seq_length=nn_max_faces,
        quantization_bits=8,
        decoder_cross_attention=True,  # todo: cross attention for decoder?
        use_discrete_vertex_embeddings=True,  # todo: discrete vertex embedding?
        name=f"face_model_{i_p}"
    )
    face_models.append(cur_part_face_model)
    # cur_part_face_model_pred_dist = face_models[-1](face_model_batches[i_p]) # face_model for ...
    face_model_pred_dists.append(face_models[-1](face_model_batches[i_p]))
    # face_model_loss ---- how to get face model loss? --- faces and meshes; losses
    cur_part_face_model_loss = -tf.reduce_sum(
        face_model_pred_dists[-1].log_prob(face_model_batches[i_p]['faces']) *
        face_model_batches[i_p]['faces_mask'])
    cur_part_face_samples = face_models[-1].sample(
        context=vertex_samples[i_p], max_sample_length=nn_max_faces, top_p=0.95,
        only_return_complete=False)
    print(face_model_batches[i_p])
    # print(cur_part_face_model_pred_dist)
    print(cur_part_face_samples)

    face_samples.append(cur_part_face_samples)
    face_model_losses.append(cur_part_face_model_loss)

# Optimization settings
learning_rate = 5e-4
training_steps = 500
check_step = 5

# Create an optimizer an minimize the summed log probability of the mesh
# sequences
optimizer = tf.train.AdamOptimizer(learning_rate)
vertex_model_optim_ops = []
face_model_optim_ops = []
tot_vertex_model_loss, tot_face_model_loss = 0., 0.

for i_p, cur_part_vertex_model_loss in enumerate(vertex_model_losses):
    #
    # cur_part_vertex_model_optim_op = optimizer.minimize(vertex_model_losses[i_p])
    # vertex_model_optim_ops.append(cur_part_vertex_model_optim_op)

    # tot_vertex_model_loss += cur_part_vertex_model_loss

    tot_vertex_model_loss += vertex_model_losses[i_p]  # construct vertex model loss
    vertex_model_optim_ops.append(optimizer.minimize(vertex_model_losses[i_p]))  # construct vertex model optimizer

for i_p, cur_part_face_model_loss in enumerate(face_model_losses):  # face models...
    # cur_part_face_model_optim_op = optimizer.minimize(cur_part_face_model_loss)
    # face_model_optim_ops.append(cur_part_face_model_optim_op)
    # tot_face_model_loss += cur_part_face_model_loss
    tot_face_model_loss += face_model_losses[i_p]  # face model loss
    face_model_optim_ops.append(optimizer.minimize(face_model_losses[i_p]))  # construct face model optimization op

print("start training")
# Training loop
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for n in range(training_steps):
        if n % check_step == 0:
            # v_loss, f_loss = sess.run((vertex_model_loss, face_model_loss)) # session_run(vertex_model_loss, face_model_loss)
            v_loss, f_loss = sess.run(
                (tot_vertex_model_loss, tot_face_model_loss))  # run the vertex loss and face loss for loss computing
            print('Step {}'.format(n))
            print('Loss (vertices) {}'.format(v_loss))
            print('Loss (faces) {}'.format(f_loss))
            # vertex-model-batch...; samples_np...;
            # v_samples_np, f_samples_np, b_np = sess.run(
            # (vertex_samples, face_samples, vertex_model_batch)) # give vertex_model_batch (context information) for further sampling
            tot_mesh_list = []
            tot_n_samples = 4
            tot_n_part = len(vertex_samples)  # n_part # n_vertices
            for i_p in range(len(vertex_samples)):  # n_part
                cur_part_v_samples_np, cur_part_f_samples_np, cur_part_b_np = sess.run(
                    (vertex_samples[i_p], face_samples[i_p], vertex_model_batches[i_p])  #
                )
                mesh_list = []  # mesh list
                for n in range(4):
                    mesh_list.append(
                        {
                            'vertices': cur_part_v_samples_np['vertices'][n][:cur_part_v_samples_np['num_vertices'][n]],
                            # num_vertices, faces
                            'faces': data_utils.unflatten_faces(
                                cur_part_f_samples_np['faces'][n][:cur_part_f_samples_np['num_face_indices'][n]])
                        }
                    )
                data_utils.plot_meshes(mesh_list, ax_lims=0.5)  # plot meshes in the mesh list
                tot_mesh_list.append(mesh_list)

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
                    print(f"i_s: {i_s}, i_p: {i_p}, n_vertices: {cur_s_cur_part_vertices.shape[0]}")
                    cur_s_tot_faces += cur_s_cur_part_new_faces

                cur_s_tot_vertices = np.concatenate(cur_s_tot_vertices, axis=0)
                print(f"i_s: {i_s}, n_cur_s_tot_vertices: {cur_s_tot_vertices.shape[0]}")
                cur_s_mesh_dict = {
                    'vertices': cur_s_tot_vertices, 'faces': cur_s_tot_faces
                }
                tot_samples_mesh_dict.append(cur_s_mesh_dict)
            data_utils.plot_meshes(tot_samples_mesh_dict, ax_lims=0.5)  # plot the mesh;

        # sess.run((vertex_model_optim_op, face_model_optim_op)) # face_model_optim_op
        sess.run((vertex_model_optim_ops[0], face_model_optim_ops[0], vertex_model_optim_ops[1],
                  face_model_optim_ops[1]))  # face_model_optim_op