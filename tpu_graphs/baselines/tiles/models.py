# Copyright 2023 The tpu_graphs Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines GNNs and MLP models for ranking module configurations on tiles data.

The high-level models are:
  + LateJoinResGCN: Applies GNN on op nodes. The GNN output will be concatenated
    with module config features. Finally, MLP outputs scalar that ranks each
    config. Here, GNN is GCN with residual connections.
  + EarlyJoinResGCN: Like above, however, it replicates (==broadcasts) module
    config features on op nodes then applies ResGCN, then applies MLP.
  + EarlyJoinSAGE and LateJoinSAGE: like above, but using GraphSAGE as backbone.

[GCN] Kipf and Welling, ICLR'17.
[GraphSAGE] Hamilton et al, NeurIPS'17.
"""
import abc, tqdm
import numpy as np

import xgboost as xgb

import tensorflow as tf
import tensorflow_gnn as tfgnn

from tpu_graphs.baselines.tiles import implicit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
# from spektral.layers import GATConv


class _ConfigFeatureJoiner(abc.ABC):
  """Defines interface for joining config features with op nodes.

  The implementations join features pre- or post-GNN, respectively, named as
  `_EarlyJoin` and `_LateJoin`.
  """

  @abc.abstractmethod
  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    """Should return feature matrix (or tensor) of op-nodes."""
    raise NotImplementedError()

  def get_penultimate_output(
      self, pooled: tf.Tensor, unused_graph: tfgnn.GraphTensor,
      unused_num_configs: int) -> tf.Tensor:
    """Must return tensor with shape `[batch_size, num_configs, hidden_dim]`."""
    return pooled


def _mlp(dims, hidden_activation, l2reg=1e-4, use_bias=True):
  """Helper function for multi-layer perceptron (MLP)."""
  layers = []
  for i, dim in enumerate(dims):
    if i > 0:
      layers.append(tf.keras.layers.Activation(hidden_activation))
    layers.append(tf.keras.layers.Dense(
        dim, kernel_regularizer=tf.keras.regularizers.l2(l2reg),
        use_bias=use_bias))
  return tf.keras.Sequential(layers)


class _OpEmbedding(tf.keras.Model):
  """Embeds GraphTensor.node_sets['op']['op'] nodes into feature 'op_e'."""

  def __init__(self, num_ops: int, embed_d: int, l2reg: float = 1e-4):
    super().__init__()
    self.embedding_layer = tf.keras.layers.Embedding(
        num_ops, embed_d, activity_regularizer=tf.keras.regularizers.l2(l2reg))

  def call(
      self, graph: tfgnn.GraphTensor,
      training: bool = False) -> tfgnn.GraphTensor:
    op_features = dict(graph.node_sets['op'].features)
    op_features['op_e'] = self.embedding_layer(
        tf.cast(graph.node_sets['op']['op'], tf.int32))
    return graph.replace_features(node_sets={'op': op_features})


class ResidualGNNLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, activation_fn):
        super(ResidualGNNLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(hidden_dim)
        self.projection = tf.keras.layers.Dense(hidden_dim)  # Projection layer
        self.activation_fn = activation_fn

    def call(self, inputs, adj_matrix):
        x = inputs
        y = adj_matrix @ x
        y = self.dense(y)
        y = self.activation_fn(y)

        x_projected = self.projection(x)  # Project x to match y's dimension
        return x_projected + y


class _SAGE(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GraphSAGE GNN Backbone."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, final_mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 64):
    super().__init__()
    self._num_configs = num_configs     # 10, 'Number of configurations to consider in ranked-list.')
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim) # (108, 64)
    self._gnn_layers = []

    self._postnet = _mlp(
        [hidden_dim] * final_mlp_layers + [1], hidden_activation)
    self._activation_fn = getattr(tf.nn, hidden_activation)
    
    for _ in range(num_gnns):
        self._gnn_layers.append(ResidualGNNLayer(hidden_dim, self._activation_fn))

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)

    tf.print("x shape before is", tf.shape(x))

    bidirectional_adj = implicit.AdjacencyMultiplier(graph, 'feed')
    bidirectional_adj = implicit.Sum(
        bidirectional_adj, bidirectional_adj.transpose())

    # for gnn_layer in self._gnn_layers:
    #   y = bidirectional_adj @ x
    #   y = tf.concat([y, x], axis=-1)
    #   y = gnn_layer(y)
    #   y = self._activation_fn(y)
    #   y = tf.nn.l2_normalize(y, axis=-1)
    #   x = y
      
    for gnn_layer in self._gnn_layers:
      x = gnn_layer(x, bidirectional_adj)
      x = tf.nn.l2_normalize(x, axis=-1)

    tf.print("x shape after is", tf.shape(x))  

    pooled = tfgnn.pool_nodes_to_context(graph, 'op', 'sum', feature_value=x)
    tf.print("pooled shape is", tf.shape(pooled))

    pooled = self.get_penultimate_output(pooled, graph, num_configs)
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    # _postnet maps across last channel from hidden_dim to 1.
    return tf.squeeze(self._postnet(pooled), -1)


class _ResGCN(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GCN backbone with residual connections."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 32, directed: bool = False,
               reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._gc_layers = []
    self._activation_fn = getattr(tf.nn, hidden_activation)
    self._directed = directed
    self._reduction = reduction
    self._prenet = _mlp([hidden_dim, hidden_dim], self._activation_fn)
    self._postnet = _mlp([hidden_dim, 1], self._activation_fn)
    for _ in range(num_gnns):
      if directed:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn),
                        _mlp([hidden_dim] * mlp_layers, self._activation_fn))
      else:
        configs_mlps = (_mlp([hidden_dim] * mlp_layers, self._activation_fn),)
      self._gc_layers.append(tuple(configs_mlps))

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)

    am = implicit.AdjacencyMultiplier(graph, 'feed')
    am = am.add_eye().normalize_right()
    x = self._prenet(x)
    for gc_layer in self._gc_layers:
      y = self._activation_fn(x)
      forward_layer = gc_layer[0]
      if self._directed:
        reverse_layer = gc_layer[1]
        self_layer = gc_layer[2]
        y = (forward_layer(am @ y) + reverse_layer(am.transpose() @ y)
             + self_layer(y))
      else:
        y = forward_layer((am @ y) + (am.transpose() @ y)  + y)

      # Residual connection.
      x += y

    x = self._activation_fn(x)
    pooled = tfgnn.pool_nodes_to_context(
        graph, 'op', self._reduction, feature_value=x)
    # Pooled has shape [batch_size, num_configs, hidden_dim]

    pooled = self.get_penultimate_output(pooled, graph, num_configs)

    return tf.squeeze(self._postnet(pooled), -1)


class _EarlyJoin(_ConfigFeatureJoiner):
  """Joins module configuration features before applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tf.Tensor:
    graph = _EarlyJoin.attach_config_features_on_op_nodes(graph)
    return tf.concat([
        # Shape (num_nodes, num_configs, embedding dim)
        tf.stack([graph.node_sets['op']['op_e']] * num_configs, 1),
        # Shape (num_nodes, num_configs, config feat dim)
        graph.node_sets['op']['config_feats'],
        # Shape (num_nodes, num_configs, op feat dim)
        tf.stack([graph.node_sets['op']['feats']] * num_configs, 1),
    ], axis=-1)

  @staticmethod
  def attach_config_features_on_op_nodes(
      graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """Replicates config features on every op node."""
    # Shape: [batch_size * num_configs, feature size].
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # shape: (total number of op nodes, config feats dimension)
    op_broadcasted = tfgnn.broadcast_node_to_edges(
        graph, 'g_op', tfgnn.SOURCE, feature_value=config_feats)
    op_features = dict(graph.node_sets['op'].features)
    op_features['config_feats'] = op_broadcasted
    return graph.replace_features(node_sets={'op': op_features})


class _LateJoin(_ConfigFeatureJoiner):
  """Joins module configuration features after applying GNN backbone."""

  def get_op_node_features(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    del num_configs
    return tf.concat([
        # Shape (num_nodes, embedding dim)
        graph.node_sets['op']['op_e'],
        # Shape (num_nodes, op feat dim)
        graph.node_sets['op']['feats'],
    ], axis=-1)

  def get_penultimate_output(
      self, pooled: tf.Tensor, graph: tfgnn.GraphTensor,
      num_configs: int) -> tf.Tensor:
    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    pooled = tf.stack([pooled] * num_configs, 1)
    pooled = tf.concat([pooled, config_feats], -1)
    return pooled



class _CombinedGNNModel(tf.keras.Model, _ConfigFeatureJoiner):
    """Combined Graph Neural Network Model integrating features of _SAGE and _ResGCN."""
    def __init__(self, num_configs: int, num_ops: int, num_gnns: int = 3,
                 mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
                 hidden_dim: int = 64, op_embed_dim: int = 64,
                 directed: bool = False, reduction: str = 'sum'):
        super().__init__()
        self._num_configs = num_configs
        self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
        self._activation_fn = getattr(tf.nn, hidden_activation)
        self._directed = directed
        self._reduction = reduction

        # Shared MLP for both models
        self._shared_mlp = _mlp([hidden_dim] * mlp_layers, self._activation_fn)

        # GNN layers from _SAGE
        self._gnn_layers = []
        for _ in range(num_gnns):
            self._gnn_layers.append(_mlp([hidden_dim], hidden_activation))

        # GC layers from _ResGCN
        self._gc_layers = []
        for _ in range(num_gnns):
            if directed:
                configs_mlps = (self._shared_mlp, self._shared_mlp, self._shared_mlp)
            else:
                configs_mlps = (self._shared_mlp,)
            self._gc_layers.append(tuple(configs_mlps))

        # Combining post-nets from both models
        self._postnet = _mlp([hidden_dim, hidden_dim, 1], self._activation_fn)

    def call(self, graph: tfgnn.GraphTensor, training: bool = False):
        graph = self._op_embedding(graph)
        x = self.get_op_node_features(graph, self._num_configs)

        # Apply _SAGE GNN layers
        for gnn_layer in self._gnn_layers:
            x = gnn_layer(x)

        # Apply _ResGCN GC layers with residual connections
        am = implicit.AdjacencyMultiplier(graph, 'feed').add_eye().normalize_right()
        for gc_layer in self._gc_layers:
            y = self._activation_fn(x)
            forward_layer = gc_layer[0]
            if self._directed:
                reverse_layer = gc_layer[1]
                self_layer = gc_layer[2]
                y = (forward_layer(am @ y) + reverse_layer(am.transpose() @ y) + self_layer(y))
            else:
                y = forward_layer((am @ y) + (am.transpose() @ y) + y)
            x += y  # Residual connection

        x = self._activation_fn(x)
        pooled = tfgnn.pool_nodes_to_context(graph, 'op', self._reduction, feature_value=x)
        
        return tf.squeeze(self._postnet(pooled), -1)
    def forward(self, graph: tfgnn.GraphTensor, num_configs):
        graph = self._op_embedding(graph)
        x = self.get_op_node_features(graph, num_configs)

        # Apply _SAGE GNN layers
        for gnn_layer in self._gnn_layers:
            x = gnn_layer(x)

        # Apply _ResGCN GC layers with residual connections
        am = implicit.AdjacencyMultiplier(graph, 'feed').add_eye().normalize_right()
        for gc_layer in self._gc_layers:
            y = self._activation_fn(x)
            forward_layer = gc_layer[0]
            if self._directed:
                reverse_layer = gc_layer[1]
                self_layer = gc_layer[2]
                y = (forward_layer(am @ y) + reverse_layer(am.transpose() @ y) + self_layer(y))
            else:
                y = forward_layer((am @ y) + (am.transpose() @ y) + y)
            x += y  # Residual connection

        x = self._activation_fn(x)
        pooled = tfgnn.pool_nodes_to_context(graph, 'op', self._reduction, feature_value=x)
        
        return tf.squeeze(self._postnet(pooled), -1)

class EarlyJoinCombinedGNNModel(_EarlyJoin, _CombinedGNNModel):
  pass


class LateJoinResGCN(_LateJoin, _ResGCN):
  pass


class EarlyJoinResGCN(_EarlyJoin, _ResGCN):
  pass


class LateJoinSAGE(_LateJoin, _SAGE):
  pass


class EarlyJoinSAGE(_EarlyJoin, _SAGE):
  pass


class _SAGE_Extractor(tf.keras.Model, _ConfigFeatureJoiner):
  """Implements GraphSAGE GNN Backbone."""

  def __init__(self, num_configs: int, num_ops: int,
               num_gnns: int = 3, final_mlp_layers: int = 2,
               hidden_activation: str = 'leaky_relu', hidden_dim: int = 64,
               op_embed_dim: int = 64):
    super().__init__()
    self._num_configs = num_configs     # 10, 'Number of configurations to consider in ranked-list.')
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim) # (108, 64)
    self._gnn_layers = []
    for unused_i in range(num_gnns):
      self._gnn_layers.append(_mlp([hidden_dim], hidden_activation))
    self._postnet = _mlp(
        [hidden_dim] * final_mlp_layers + [1], hidden_activation)
    self._activation_fn = getattr(tf.nn, hidden_activation)
    tf.print("SAGE_Extractor")

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    pooled = self.forward(graph, self._num_configs)
    return tf.squeeze(self._postnet(pooled), -1)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    x = self.get_op_node_features(graph, num_configs)
    bidirectional_adj = implicit.AdjacencyMultiplier(graph, 'feed')
    bidirectional_adj = implicit.Sum(
        bidirectional_adj, bidirectional_adj.transpose())
    for gnn_layer in self._gnn_layers:
      y = bidirectional_adj @ x
      y = tf.concat([y, x], axis=-1)
      y = gnn_layer(y)
      y = self._activation_fn(y)
      y = tf.nn.l2_normalize(y, axis=-1)
      x = y

    pooled = tfgnn.pool_nodes_to_context(graph, 'op', 'sum', feature_value=x)

    pooled = self.get_penultimate_output(pooled, graph, num_configs)
    return pooled
    # Pooled has shape [batch_size, num_configs, hidden_dim]
    # _postnet maps across last channel from hidden_dim to 1.
    return tf.squeeze(self._postnet(pooled), -1)


class EarlyJoinSAGEExtractor(_EarlyJoin, _SAGE_Extractor):
  pass


class MLP(tf.keras.Model):
  """Embeds op codes, averages features across all-nodes, passing thru MLP."""

  def __init__(
      self, num_configs: int, num_ops: int, op_embed_dim: int = 32,
      mlp_layers: int = 2, hidden_activation: str = 'leaky_relu',
      hidden_dim: int = 64, reduction: str = 'sum'):
    super().__init__()
    self._num_configs = num_configs
    self._num_ops = num_ops
    self._op_embedding = _OpEmbedding(num_ops, op_embed_dim)
    self._reduction = reduction
    layer_dims = [hidden_dim] * mlp_layers
    layer_dims.append(1)
    self._mlp = _mlp(layer_dims, hidden_activation)

  def call(self, graph: tfgnn.GraphTensor, training: bool = False):
    del training
    return self.forward(graph, self._num_configs)

  def forward(
      self, graph: tfgnn.GraphTensor, num_configs: int) -> tfgnn.GraphTensor:
    graph = self._op_embedding(graph)
    op_feats = tf.concat([
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='feats'),
        tfgnn.pool_nodes_to_context(
            graph, 'op', self._reduction, feature_name='op_e'),
    ], axis=-1)

    config_feats = graph.node_sets['config']['feats']
    batch_size = graph.node_sets['config'].sizes.shape[0]
    config_feats = tf.reshape(
        config_feats, [batch_size, -1, config_feats.shape[-1]])
    # Shape like config feats
    op_feats = tf.stack([op_feats] * num_configs, 1)
    op_feats = tf.concat([op_feats, config_feats], -1)
    return tf.squeeze(self._mlp(op_feats), -1)



class LinearRegressor(tf.keras.Model):
    """Linear regression model for GraphTensor data."""

    def __init__(self, train_ds: tf.data.Dataset):
      super().__init__()
      # self.regressor = LinearRegression()
      # self.regressor = RandomForestRegressor(n_jobs=-1, verbose=2)

      self.param_grid = {
          'n_estimators': [100, 200, 300],
          'learning_rate': [0.01, 0.1, 0.2],
          'max_depth': [3, 4, 5],
          # Add more parameters here
      }
      
      self.regressor = xgb.XGBRegressor(objective='rank:pairwise', n_jobs=48,
                                        n_estimators=100,
                                        learning_rate=0.01,
                                        )
      self.best_regressor = None
      self.train(train_ds)

    def train(self, train_ds):
      X = []
      y = []
      for graph in tqdm.tqdm(train_ds):
          runtimes = graph.node_sets['config']['runtimes']
          config_feat=graph.node_sets['config']['feats']
          X.append(config_feat.numpy())
          y.append(runtimes.numpy())
      X = np.concatenate(X)
      y = np.concatenate(y)

      self.regressor.fit(X, y)
      print("training finished")
      
      with open('model.pkl', 'wb') as file:
          pickle.dump(self.regressor, file)
          
      # grid_search = GridSearchCV(
      #     estimator=self.regressor,
      #     param_grid=self.param_grid,
      #     cv=3,  # Number of cross-validation folds
      #     scoring='neg_mean_squared_error',  # Define your scoring metric
      #     verbose=2
      # )

      # grid_search.fit(X, y)
      # self.best_regressor = grid_search.best_estimator_  # The best model

      # print("Best parameters found: ", grid_search.best_params_)
      # print("Best score: ", grid_search.best_score_)

    def call(self, graph: tfgnn.GraphTensor, training: bool = False):
      del training
      return self.forward(graph, self._num_configs)

    def forward(self, graph: tfgnn.GraphTensor, num_configs: int):
      # Extracting features as done in the MLP model
      # You may need to adjust this part based on how your graph's features are structured

      config_feat=graph.node_sets['config']['feats']
      # pred_time = self.regressor.predict(config_feat)
      pred_time = self.regressor.predict(config_feat)
      return tf.reshape(pred_time, (1, -1))


class Extractor_Linear():
    def __init__(self, X, y, X_val, y_val):
      super().__init__()
      self.bst = None
      self.X = X
      self.y = y
      self.X_val = X_val
      self.y_val = y_val
      self.trained = False

    def train(self):
      dtrain = xgb.DMatrix(self.X, label=self.y)
      dvalid = xgb.DMatrix(self.X_val, label=self.y_val)
      params = {
          'learning_rate': 0.01,
          'max_depth': 3,
          'objective': 'reg:squarederror',
          'n_jobs': 48,
          'seed': 42,
      }
      
      self.bst = xgb.train(params, dtrain, num_boost_round=100, 
                          evals=[(dtrain, 'train'), (dvalid, 'valid')],
                          early_stopping_rounds=10)
      
      print("training finished")
      
      self.bst.save_model('xgb_model.json')
      self.trained = True

    def forward(self, X):
      # Extracting features as done in the MLP model
      # You may need to adjust this part based on how your graph's features are structured
      if not self.trained:
        # load model
        self.bst = xgb.Booster()
        self.bst.load_model('xgb_model.json')
        self.trained = True 
        
      if X.ndim == 1:
          X = X.reshape(1, -1)
      dmat = xgb.DMatrix(X)

      y = self.bst.predict(dmat)
      return y