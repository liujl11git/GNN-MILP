import tensorflow as tf
import tensorflow.keras as K
import numpy as np
import pickle


class BipartiteGraphConvolution(K.Model):
	"""
	Partial bipartite graph convolution (either left-to-right or right-to-left).
	"""

	def __init__(self, emb_size, activation, initializer, right_to_left=False):
		super().__init__()
		
		self.emb_size = emb_size
		self.activation = activation
		self.initializer = initializer
		self.right_to_left = right_to_left

		# feature layers
		self.feature_module_left = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=None, use_bias=True, kernel_initializer=self.initializer),
			K.layers.Activation(self.activation),
			K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
		])
		self.feature_module_edge = K.Sequential([
		])
		self.feature_module_right = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=None, use_bias=False, kernel_initializer=self.initializer),
			K.layers.Activation(self.activation),
			K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
		])

		# output_layers
		self.output_module = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
			K.layers.Activation(self.activation),
			K.layers.Dense(units=self.emb_size, activation=None, kernel_initializer=self.initializer),
		])
	
	def build(self, input_shapes):
		
		l_shape, ei_shape, ev_shape, r_shape = input_shapes

		self.feature_module_left.build(l_shape)
		self.feature_module_edge.build(ev_shape)
		self.feature_module_right.build(r_shape)
		self.output_module.build([None, self.emb_size + (l_shape[1] if self.right_to_left else r_shape[1])])
		self.built = True
	
	def call(self, inputs):
	
		left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

		if self.right_to_left:
			scatter_dim = 0
			prev_features = self.feature_module_left(left_features)
		else:
			scatter_dim = 1
			prev_features = self.feature_module_right(right_features)

		# compute joint features
		if scatter_dim == 0:
			joint_features = self.feature_module_edge(edge_features) * tf.gather(
					self.feature_module_right(right_features),
					axis=0,
					indices=edge_indices[1]
				)
		else:
			joint_features = self.feature_module_edge(edge_features) * tf.gather(
					self.feature_module_left(left_features),
					axis=0,
					indices=edge_indices[0]
				)

		# perform convolution
		conv_output = tf.scatter_nd(
			updates=joint_features,
			indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
			shape=[scatter_out_size, self.emb_size]
		)

		# mean convolution
		neighbour_count = tf.scatter_nd(
			updates=tf.ones(shape=[tf.shape(edge_indices)[1], 1], dtype=tf.float32),
			indices=tf.expand_dims(edge_indices[scatter_dim], axis=1),
			shape=[scatter_out_size, 1])
		output = self.output_module(tf.concat([
			conv_output,
			prev_features,
		], axis=1))

		return output

class GCNPolicy(K.Model):
	"""
	Our bipartite Graph Convolutional neural Network (GCN) model.
	"""

	def __init__(self, embSize, nConsF, nEdgeF, nVarF, isGraphLevel = True):
		super().__init__()

		self.emb_size = embSize
		self.cons_nfeats = nConsF
		self.edge_nfeats = nEdgeF
		self.var_nfeats = nVarF
		self.is_graph_level = isGraphLevel 
		# "isGraphLevel == True" means the output is graph-level, each graph has an output value;
		# Otherwise, each variable has an output value.

		self.activation = K.activations.relu
		self.initializer = K.initializers.Orthogonal()

		# CONSTRAINT EMBEDDING
		self.cons_embedding = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
		])

		# EDGE EMBEDDING
		self.edge_embedding = K.Sequential([
		])

		# VARIABLE EMBEDDING
		self.var_embedding = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
		])

		# GRAPH CONVOLUTIONS
		self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
		self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)

		# GRAPH CONVOLUTIONS
		self.conv_v_to_c2 = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
		self.conv_c_to_v2 = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)

		# OUTPUT
		self.output_module = K.Sequential([
			K.layers.Dense(units=self.emb_size, activation=self.activation, kernel_initializer=self.initializer),
			K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer, use_bias=False),
		])
		
		# build model right-away
		self.build([
			(None, self.cons_nfeats),
			(2, None),
			(None, self.edge_nfeats),
			(None, self.var_nfeats),
			(None, ),
			(None, ),
		])
		
		# save / restore fix
		self.variables_topological_order = [v.name for v in self.variables]
		
		# save input signature for compilation
		self.input_signature = [
			(
				tf.TensorSpec(shape=[None, self.cons_nfeats], dtype=tf.float32),
				tf.TensorSpec(shape=[2, None], dtype=tf.int32),
				tf.TensorSpec(shape=[None, self.edge_nfeats], dtype=tf.float32),
				tf.TensorSpec(shape=[None, self.var_nfeats], dtype=tf.float32),
				tf.TensorSpec(shape=[None], dtype=tf.int32),
				tf.TensorSpec(shape=[None], dtype=tf.int32),
			),
			tf.TensorSpec(shape=[], dtype=tf.bool),
		]
		
	
	def build(self, input_shapes):
		
		c_shape, ei_shape, ev_shape, v_shape, nc_shape, nv_shape = input_shapes
		emb_shape = [None, self.emb_size]

		if not self.built:
			self.cons_embedding.build(c_shape)
			self.edge_embedding.build(ev_shape)
			self.var_embedding.build(v_shape)
			self.conv_v_to_c.build((emb_shape, ei_shape, emb_shape, emb_shape))
			self.conv_c_to_v.build((emb_shape, ei_shape, emb_shape, emb_shape))
			self.conv_v_to_c2.build((emb_shape, ei_shape, emb_shape, emb_shape))
			self.conv_c_to_v2.build((emb_shape, ei_shape, emb_shape, emb_shape))
			if self.is_graph_level:
				self.output_module.build([None, 2 * self.emb_size])
			else:
				self.output_module.build(emb_shape)
			self.built = True

	def call(self, inputs, training):
		
		constraint_features, edge_indices, edge_features, variable_features, n_cons_total, n_vars_total, n_cons_small, n_vars_small = inputs

		# EMBEDDINGS
		constraint_features = self.cons_embedding(constraint_features)
		edge_features = self.edge_embedding(edge_features)
		variable_features = self.var_embedding(variable_features)

		# GRAPH CONVOLUTIONS
		constraint_features = self.conv_v_to_c((
			constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
		constraint_features = self.activation(constraint_features)

		variable_features = self.conv_c_to_v((
			constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
		variable_features = self.activation(variable_features)

		constraint_features = self.conv_v_to_c2((
			constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
		constraint_features = self.activation(constraint_features)

		variable_features = self.conv_c_to_v2((
			constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
		variable_features = self.activation(variable_features)
		
		if self.is_graph_level:
			variable_features = tf.reshape(variable_features, [int(n_vars_total / n_vars_small), n_vars_small, self.emb_size])
			variable_features_mean = tf.reduce_mean(variable_features, axis = 1)
			constraint_features = tf.reshape(constraint_features, [int(n_cons_total / n_cons_small), n_cons_small, self.emb_size])
			constraint_features_mean = tf.reduce_mean(constraint_features, axis = 1)
			final_features = tf.concat([variable_features_mean, constraint_features_mean], 1)
		else:
			final_features = variable_features

		# OUTPUT
		output = self.output_module(final_features)
		return output
		
	def save_state(self, path):
		with open(path, 'wb') as f:
			for v_name in self.variables_topological_order:
				v = [v for v in self.variables if v.name == v_name][0]
				pickle.dump(v.numpy(), f)

	def restore_state(self, path):
		with open(path, 'rb') as f:
			for v_name in self.variables_topological_order:
				v = [v for v in self.variables if v.name == v_name][0]
				v.assign(pickle.load(f))


