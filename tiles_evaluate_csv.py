from tpu_graphs.baselines.tiles import data
from tpu_graphs.baselines.tiles import train_lib

import os, tqdm, collections, json
import tensorflow as tf
from absl import flags, app

_MODEL_NAME = flags.DEFINE_string(
    'name', None, 'Model name ', required=True,)

def main(unused_argv: list[str]):

    f_out = open("./predictions/perf.log", "a")

    dirpaths = [os.path.join("./predictions", f) for f in os.listdir("./predictions") if f.endswith('.csv')]

    for result_path in dirpaths:
        if _MODEL_NAME.value is not None and _MODEL_NAME.value not in result_path:
            continue
        dict = {}

        with open(result_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                ID, TopConfigs = line.strip().split(",")
                TopConfigs = TopConfigs.split(";")
                ID = ID.split(":")[-1]
                dict[ID] = TopConfigs
                
        dataset = data.get_npz_dataset(
            os.path.expanduser(train_lib._DATA_ROOT.value),
            cache_dir=os.path.expanduser(train_lib._CACHE_DIR.value))
        ds = dataset.validation.get_graph_tensors_dataset()
        
        errors = []
        errors_by_benchmark = collections.defaultdict(list)
        for graph in tqdm.tqdm(ds):
            # num_configs = int(graph.node_sets['config'].sizes[0])
            module_id = graph.node_sets['g']['tile_id'][0].numpy().decode('utf-8')
            benchmark = (graph.node_sets['g']['tile_id'].numpy()[0].decode().rsplit('_', 1)[0])
            
            runtimes = graph.node_sets['config']['runtimes']
            # features = graph.node_sets['config']['feats']
            time_best = tf.reduce_min(runtimes)

            if module_id not in dict:
                print("Module not found in predictions")
                continue
            
            TopConfigs = list(map(int, dict[module_id]))
            time_model_candidates = tf.gather(runtimes, TopConfigs)
            best_of_candidates = tf.reduce_min(time_model_candidates)
            error = float((best_of_candidates - time_best) / time_best)
            errors_by_benchmark[benchmark].append(error)
            errors.append(error)
            
        filename = os.path.basename(result_path)
        errors_data = {k2: float(tf.reduce_mean(v2).numpy()) for k2, v2 in errors_by_benchmark.items()}
        average_error = float(tf.reduce_mean(errors).numpy())
        errors_data['average'] = average_error
        output_data = {'filename': filename}
        output_data.update(errors_data)
        print(json.dumps(output_data, indent=2))
        f_out.write(json.dumps(output_data, indent=2))
        
    f_out.close()
            


if __name__ == '__main__':
  app.run(main)