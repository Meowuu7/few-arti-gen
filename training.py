import os
import numpy as np
import tensorflow.compat.v1 as tf
# tf.logging.set_verbosity(tf.logging.ERROR)  # Hide TF deprecation messages
import matplotlib.pyplot as plt

import modules
import data_utils

# Prepare synthetic dataset
ex_list = []
for k, mesh in enumerate(['cube', 'cylinder', 'cone', 'icosphere']):
  mesh_dict = data_utils.load_process_mesh(
      os.path.join('meshes', '{}.obj'.format(mesh)))
  mesh_dict['class_label'] = k
  ex_list.append(mesh_dict)
synthetic_dataset = tf.data.Dataset.from_generator(
    lambda: ex_list,
    output_types={
        'vertices': tf.int32, 'faces': tf.int32, 'class_label': tf.int32},
    output_shapes={
        'vertices': tf.TensorShape([None, 3]), 'faces': tf.TensorShape([None]),
        'class_label': tf.TensorShape(())}
    )
ex = synthetic_dataset.make_one_shot_iterator().get_next()

# Inspect the first mesh
with tf.Session() as sess:
  ex_np = sess.run(ex)
print(ex_np)

# Plot the meshes
mesh_list = []
with tf.Session() as sess:
  for i in range(4):
    ex_np = sess.run(ex)
    mesh_list.append(
        {'vertices': data_utils.dequantize_verts(ex_np['vertices']),
         'faces': data_utils.unflatten_faces(ex_np['faces'])})
# data_utils.plot_meshes(mesh_list, ax_lims=0.4)

### cages, field, ###
# Prepare the dataset for vertex model training
import modules
import data_utils
vertex_model_dataset = data_utils.make_vertex_model_dataset(
    synthetic_dataset, apply_random_shift=False)
vertex_model_dataset = vertex_model_dataset.repeat()
vertex_model_dataset = vertex_model_dataset.padded_batch(
    4, padded_shapes=vertex_model_dataset.output_shapes)
vertex_model_dataset = vertex_model_dataset.prefetch(1)
vertex_model_batch = vertex_model_dataset.make_one_shot_iterator().get_next()

# Create vertex model
vertex_model = modules.VertexModel(
    decoder_config={
        'hidden_size': 128,
        'fc_size': 512,
        'num_layers': 3,
        'dropout_rate': 0.
    },
    class_conditional=True, # class conditional generation
    num_classes=4,
    max_num_input_verts=250,
    quantization_bits=8,
)

# some statistics
print("vertex_model_batch", type(vertex_model_batch), vertex_model_batch.keys())
# vertex model pred dist
vertex_model_pred_dist = vertex_model(vertex_model_batch, is_training=True)
vertex_model_loss = -tf.reduce_sum(
    vertex_model_pred_dist.log_prob(vertex_model_batch['vertices_flat']) *
    vertex_model_batch['vertices_flat_mask'])
vertex_samples = vertex_model.sample(
    4, context=vertex_model_batch, max_sample_length=200, top_p=0.95,
    recenter_verts=False, only_return_complete=False)

print(vertex_model_batch)
print(vertex_model_pred_dist)
print(vertex_samples)
print("here1")


face_model_dataset = data_utils.make_face_model_dataset(
    synthetic_dataset, apply_random_shift=False)
face_model_dataset = face_model_dataset.repeat()
face_model_dataset = face_model_dataset.padded_batch(
    4, padded_shapes=face_model_dataset.output_shapes)
face_model_dataset = face_model_dataset.prefetch(1)
face_model_batch = face_model_dataset.make_one_shot_iterator().get_next()

# Create face model
face_model = modules.FaceModel(
    encoder_config={
        'hidden_size': 128,
        'fc_size': 512,
        'num_layers': 3,
        'dropout_rate': 0.
    },
    decoder_config={
        'hidden_size': 128,
        'fc_size': 512,
        'num_layers': 3,
        'dropout_rate': 0.
    },
    class_conditional=False,
    max_seq_length=500,
    quantization_bits=8,
    decoder_cross_attention=True,
    use_discrete_vertex_embeddings=True,
)
face_model_pred_dist = face_model(face_model_batch)
face_model_loss = -tf.reduce_sum(
    face_model_pred_dist.log_prob(face_model_batch['faces']) *
    face_model_batch['faces_mask'])
face_samples = face_model.sample(
    context=vertex_samples, max_sample_length=500, top_p=0.95,
    only_return_complete=False)
print(face_model_batch)
print(face_model_pred_dist)
print(face_samples)


# Optimization settings
learning_rate = 5e-4
training_steps = 500
check_step = 5

# Create an optimizer an minimize the summed log probability of the mesh
# sequences
optimizer = tf.train.AdamOptimizer(learning_rate)
vertex_model_optim_op = optimizer.minimize(vertex_model_loss)
face_model_optim_op = optimizer.minimize(face_model_loss)

# Training loop
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for n in range(training_steps):
    if n % check_step == 0:
      v_loss, f_loss = sess.run((vertex_model_loss, face_model_loss))
      print('Step {}'.format(n))
      print('Loss (vertices) {}'.format(v_loss))
      print('Loss (faces) {}'.format(f_loss))
      v_samples_np, f_samples_np, b_np = sess.run(
        (vertex_samples, face_samples, vertex_model_batch))
      mesh_list = []
      for n in range(4):
        mesh_list.append(
            {
                'vertices': v_samples_np['vertices'][n][:v_samples_np['num_vertices'][n]],
                'faces': data_utils.unflatten_faces(
                    f_samples_np['faces'][n][:f_samples_np['num_face_indices'][n]])
            }
        )
      mesh_sv_fn = "samples/samples_" + str(n) + ".png"
      data_utils.plot_meshes(mesh_list, ax_lims=0.5, mesh_sv_fn=mesh_sv_fn)
    sess.run((vertex_model_optim_op, face_model_optim_op))


