import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os
from xgboost import XGBRegressor

# IDX = [6, 7, 14, 15]

def load_data(files):
    X, y = [], []

    for filename in files:
        # print(np.load(filename, allow_pickle=True)['config_feat'])
        unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
        runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
        runtimes = unnormalised_runtime / runtime_normalisers

        output_bounds_sums = np.load(filename, allow_pickle=True)['config_feat']#[:, IDX]

        X.extend(output_bounds_sums)
        y.extend(runtimes)

    return np.array(X), np.array(y).reshape(-1, 1)


def slowdown(y_true, y_pred, k):
    y_fastest = np.min(y_true)

    top_k_indices = y_pred[:k]

    top_k_predicted_times = np.min(y_true[top_k_indices])

    ratio = top_k_predicted_times / y_fastest

    return ratio - 1

def speed_score(y_true, y_pred, k):
    return 1 - slowdown(y_true, y_pred, k)


BASE_DIR = '/home/yuhaoge2/data/tpugraphs/npz/tile/xla/'
TRAIN_DIR = BASE_DIR + 'train/'
VALID_DIR = BASE_DIR + 'valid/'
TEST_DIR = BASE_DIR + 'test/'

np.random.seed(42)
train_files = [TRAIN_DIR + f for f in os.listdir(TRAIN_DIR)]
valid_files = [VALID_DIR + f for f in os.listdir(VALID_DIR)]

X, y = load_data(train_files)

print('X shape', X.shape)
print('y shape', y.shape)

# model = LinearRegression()
model = XGBRegressor(objective ='reg:squarederror')

model.fit(X, y)
# print("Model coefficients:", model.coef_)
# print("Model intercept:", model.intercept_)


# compare linear regression vs uniform mean prediction for validation set
train_mse = mean_squared_error(y, model.predict(X))
print("MSE on train set:", train_mse)

X_val, y_val = load_data(valid_files)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print("Validation Mean Squared Error:", mse)

mean_baseline_mse = mean_squared_error(y_val, np.mean(y_val) * np.ones_like(y_val))
print("Validation MSE for mean prediction", mean_baseline_mse)

# compare linear regression vs random for predicting runtime order

perfect_scores = []
rand_scores = []
linear_scores = []

for filename in valid_files:
    unnormalised_runtime = np.load(filename, allow_pickle=True)['config_runtime']
    runtime_normalisers = np.load(filename, allow_pickle=True)['config_runtime_normalizers']
    runtimes = unnormalised_runtime / runtime_normalisers

    perfect_preds = np.argsort(runtimes)
    perfect_score = speed_score(runtimes, perfect_preds, 3)
    perfect_scores.append(perfect_score)
    
    # randomize the indices
    np.random.shuffle(perfect_preds)
    rand_score = speed_score(runtimes, perfect_preds, 3)
    rand_scores.append(rand_score)
    
    predicted_runtimes = model.predict(np.load(filename, allow_pickle=True)['config_feat'])#[:, IDX])
    # predicted_idx = np.argsort(predicted_runtimes[:, 0])
    # print(predicted_runtimes.shape)
    predicted_idx = np.argsort(predicted_runtimes)
    linear_score = speed_score(runtimes, predicted_idx, 3)
    linear_scores.append(linear_score)

print("perfect score", np.mean(perfect_scores))
print("random score", np.mean(rand_scores))
print("linear score", np.mean(linear_scores))