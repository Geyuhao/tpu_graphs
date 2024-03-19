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

# Copyright 2023 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for invoking the training loop."""

from collections.abc import Sequence

from absl import app

import functools
import json, tqdm, collections
import os, io, gzip
from typing import Any

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_ranking as tfr
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
from tpu_graphs.baselines.tiles import train_args
import tensorflow_addons as tfa

import numpy as np

_DATA_ROOT = flags.DEFINE_string(
    'data_root', '~/data/tpugraphs/npz/tile/xla',
    'Root directory containing dataset. It must contain subdirectories '
    '{train, test, validation}, each having many .npz files')
_CACHE_DIR = flags.DEFINE_string(
    'cache_dir', '~/data/tpugraphs/cache/tile',
    'If given, dataset tensors will be cached here for faster loading.')

_MODEL_DIRS = flags.DEFINE_string(
    'dirs', None,
    'Comma-separated list of model directories to evaluate. '
    'The per-benchmark average will be printed', required=True)


def extract(model = None, ds_name='train') -> None:
  if model is None:
    dirpaths = _MODEL_DIRS.value.split(',')
    if len(dirpaths) != 1:
      print("Please provide exactly 1 model directory in --dirs.")
      return
    dirpath = dirpaths[0]
    
    dataset = data.get_npz_dataset(
        os.path.expanduser(_DATA_ROOT.value),
        cache_dir=os.path.expanduser(_CACHE_DIR.value))
    ds = dataset.validation.get_graph_tensors_dataset()
    
    # Load keras model.
    with tf.keras.saving.custom_object_scope(
        # Model was compiled with a loss before it was saved.
        # Override `load_model` in this scope to reconstruct loss object.
        {'CombinedLoss': metrics.CombinedLoss}):
      keras_model = tf.keras.models.load_model(dirpath)

    jsonz_file = dirpath.replace('/extractor_', '/run_') + '.jsonz'
    with gzip.open(open(jsonz_file, 'rb'), 'rb') as fin:
      json_data = json.loads(fin.read().decode())
      model_name = json_data['args']['model']
      model_kwargs = json.loads(json_data['args']['model_kwargs_json'])
    model_class = getattr(models, model_name)

    # Load pythonic model.
    model = model_class(
        num_configs=json_data['args']['configs'], num_ops=dataset.num_ops,
        **model_kwargs)
    
      # Instantiate `model`` parameters (to copy from `keras_model`).
    sample_graph, = ds.take(1)  # Example graph to invoke `model.forward`.
    num_configs = int(sample_graph.node_sets['config'].sizes[0])
    model.forward(sample_graph, num_configs)
    del sample_graph  # No longer need a toy example.

    target_vars = model.trainable_variables
    source_vars = keras_model.trainable_variables
    assert len(target_vars) == len(source_vars)
    for tv, sv in zip(target_vars, source_vars):
      assert sv.shape == tv.shape
      tv.assign(sv)
    print("Feature extractor loaded")
  
  if ds_name == 'train':
      dataset = data.get_npz_dataset(
          os.path.expanduser(_DATA_ROOT.value),
          cache_dir=os.path.expanduser(_CACHE_DIR.value))
      ds = dataset.train.get_graph_tensors_dataset()
  elif ds_name == 'test':
      dataset = data.get_npz_dataset(
          os.path.expanduser(_DATA_ROOT.value),
          cache_dir=os.path.expanduser(_CACHE_DIR.value))
      ds = dataset.test.get_graph_tensors_dataset()
  else:
      dataset = data.get_npz_dataset(
          os.path.expanduser(_DATA_ROOT.value),
          cache_dir=os.path.expanduser(_CACHE_DIR.value))
      ds = dataset.validation.get_graph_tensors_dataset()
      
  X = []
  y = []
  G_id = []

  for graph in tqdm.tqdm(ds):
    num_configs = int(graph.node_sets['config'].sizes[0])
    feature_emb = model.forward(graph, num_configs)
    feature_emb_processed = np.squeeze(feature_emb.numpy())
    
    runtimes = graph.node_sets['config']['runtimes']
    config_feat=graph.node_sets['config']['feats']

    combined_features = np.concatenate([feature_emb_processed, config_feat], axis=1)

    
    id = graph.node_sets['g']['tile_id'][0].numpy().decode('utf-8')
    
    id_array = np.full(runtimes.shape, id)
    
    # print(id_array.shape)
    # print(runtimes.numpy().shape)
    # print(feature_emb_processed.shape)
    X.append(combined_features)
    y.append(runtimes.numpy())
    G_id.append(id_array)
    
    # print('feature_emb', feature_emb_processed.shape)  
    # print('runtimes', runtimes.numpy().shape)  

  X = np.concatenate(X)
  y = np.concatenate(y)
  G_id = np.concatenate(G_id)
  
  # store data
  if ds_name == 'train':
      np.savez_compressed('train_data.npz', X=X, y=y, G_id=G_id)
      print("Feature extractor training data saved")
  elif ds_name == 'test':
      np.savez_compressed('test_data.npz', X=X, y=y, G_id=G_id)
      print("Feature extractor test data saved")
  else:
      np.savez_compressed('val_data.npz', X=X, y=y, G_id=G_id)
      print("Feature extractor validation data saved")

def train_regression() -> None:
  # Load data
  print("Loading data...")
  train_data = np.load('train_data.npz')
  val_data = np.load('val_data.npz')
  X = train_data['X']
  y = train_data['y']
  X_val = val_data['X']
  y_val = val_data['y']
  
  model_name = "Extractor_Linear"
  model_class = getattr(models, model_name)
  
  model = model_class(X, y, X_val, y_val)
  model.train()
  
def test_regression() -> None:
  
  print("Loading data...")
  
  test_data = np.load('test_data.npz')
  val_data = np.load('val_data.npz')
  
  model_name = "Extractor_Linear"
  model_class = getattr(models, model_name)
  
  model = model_class(None, None, None, None)
  
  module_ids, ranks = rank_config_indices(val_data, model.forward, top_ranks=5)

  os.makedirs('predictions', exist_ok=True)
  write_least_runtimes_csv(f'predictions/{model_name}_predictions_model_output.csv', module_ids, ranks)

def write_least_runtimes_csv(
    out_csv_filepath: str, module_ids: tf.Tensor, ranks: tf.Tensor):
  """Writes CSV file with line `i` containing module_ids[i] and ranks[i]."""
  csv_ranks = tf.strings.join(
      tf.strings.as_string(tf.transpose(ranks)), ';')

  stack_join = lambda x, delim: tf.strings.join(tf.stack(x), delim)
  with tf.io.gfile.GFile(out_csv_filepath, 'w') as fout:
    fout.write('ID,TopConfigs\n')
    id_vector = stack_join(
        [tf.fill(module_ids.shape, 'tile:xla'), module_ids], ':')
    csv_lines = stack_join([id_vector, csv_ranks], ',')
    fout.write(stack_join(csv_lines, '\n').numpy().decode('utf-8'))
  print('\n\n   ***  Wrote', out_csv_filepath, '\n\n')

def rank_config_indices(test_data, model, top_ranks=5):
    all_sorted_indices = []
    all_module_ids = []
    X = test_data['X']
    G_id = test_data['G_id']

    feature_per_id = collections.defaultdict(list)  # Use defaultdict for automatic list initialization

    for i in range(0, len(X)):
        feature = X[i]
        id = G_id[i]
        feature_per_id[id].append(feature)

    # for graph_id, features in feature_per_id.items():
    for graph_id, features in tqdm.tqdm(feature_per_id.items(), desc='Generating Predictions'):
        # Convert features to appropriate format if necessary (e.g., np.array or tf.Tensor)
        features = np.array(features)

        # Predict
        predicted_runtimes = model(features)
        
        # Combine indices with predictions
        indexed_predictions = list(enumerate(predicted_runtimes))
        
        # Sort based on predictions and get top ranks
        sorted_indices = sorted(indexed_predictions, key=lambda x: x[1])[:top_ranks]
        padding_needed = top_ranks - len(sorted_indices)
        if padding_needed > 0:
          sorted_indices.extend([(0, float('inf'))] * padding_needed)  # Adding placeholder padding

        all_sorted_indices.append([index for index, _ in sorted_indices])
        all_module_ids.append(graph_id)

    return tf.stack(all_module_ids, axis=0), tf.stack(all_sorted_indices, axis=0)

def main(unused_argv: Sequence[str]) -> None:
  # train(train_args.get_args())
  extract(None, 'train')
  # train_regression()
  # test_regression()

if __name__ == '__main__':
  app.run(main)
