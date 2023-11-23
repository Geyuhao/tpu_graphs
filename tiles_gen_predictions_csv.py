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

"""Evaluates model on all validation graphs, grouping metrics by benchmark."""

import collections
import gzip
import json
import os

from absl import app
from absl import flags
import tensorflow as tf
# So that keras.models.load_model() can re-instantiate layers of saved model.
import tensorflow_gnn as tfgnn  # pylint: disable=unused-import.
import tensorflow_ranking as tfr  # pylint: disable=unused-import.
from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import metrics
from tpu_graphs.baselines.tiles import models
from tpu_graphs.baselines.tiles import train_lib

import pandas as pd
unused_modules = [tfr, tfgnn]

_MODEL_DIRS = flags.DEFINE_string(
    'dir', None,
    'Path for the model directory. ', required=True)


def main(unused_argv: list[str]) -> None:
  dataset = data.get_npz_dataset(
      os.path.expanduser(train_lib._DATA_ROOT.value),
      cache_dir=os.path.expanduser(train_lib._CACHE_DIR.value))
  ds = dataset.test.get_graph_tensors_dataset()

  dirpaths = _MODEL_DIRS.value.split(',')
  if len(dirpaths) != 1:
    print("Please provide exactly 1 model directory in --dirs.")
    return
  dirpath = dirpaths[0]

  # Load keras model.
  with tf.keras.saving.custom_object_scope(
      # Model was compiled with a loss before it was saved.
      # Override `load_model` in this scope to reconstruct loss object.
      {'CombinedLoss': metrics.CombinedLoss}):
    keras_model = tf.keras.models.load_model(dirpath)

  jsonz_file = dirpath.replace('/model_', '/run_') + '.jsonz'
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

  module_ids, ranks = train_lib.rank_config_indices(ds, model.forward, top_ranks=5)

  train_lib.write_least_runtimes_csv('predictions_model_output.csv', module_ids, ranks)


if __name__ == '__main__':
  app.run(main)
