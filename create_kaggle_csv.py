import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--inputcsv', default='predictions_model_output.csv', help='The path for the .csv that contains the tile predictions from the model.')
args = parser.parse_args()

sample = pd.read_csv('sample_submission.csv')

ours = pd.read_csv(args.inputcsv)

# Replace the sample with ours only for the `tile` entries.
idx = sample['ID'].str.contains('tile')
sample.loc[idx] = ours.loc[idx]

sample.to_csv('predictions_kaggle.csv', index=False)