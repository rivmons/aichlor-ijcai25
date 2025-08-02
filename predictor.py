# # Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import os
import urllib.request

# # Suppress noisy Tensorflow debug logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TERM"] = "dumb"  # Prevents fancy terminal outputs like bars

import numpy as np
import pandas as pd
from glob import glob

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Lambda, Concatenate, Dropout
from keras.callbacks import EarlyStopping
from keras.constraints import Constraint
import keras.backend as K
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_absolute_error
import subprocess
import re

def gpu_id():
    try:
        # Run nvidia-smi and capture output
        smi_output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'])
        smi_output = smi_output.decode('utf-8').strip().split('\n')

        # Parse output into (gpu_id, mem_used) pairs
        usage = []
        for line in smi_output:
            gpu_id, mem_used = map(int, line.strip().split(','))
            usage.append((gpu_id, mem_used))

        # Find GPU with lowest memory usage
        least_used = min(usage, key=lambda x: x[1])
        return least_used[0]

    except subprocess.CalledProcessError as e:
        print("Failed to run nvidia-smi:", e)
        return None
    except Exception as e:
        print("Error while parsing nvidia-smi output:", e)
        return None

class PredictionDebugger(Callback):
    def __init__(self, model, Xc_val, Xa_val, y_val, reward_scaler, iteration=0, freq=1):
        super().__init__()
        self.Xc_val = Xc_val
        self.Xa_val = Xa_val
        self.y_val = y_val
        self.reward_scaler = reward_scaler
        self.freq = freq
        self.iteration = iteration

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            preds = self.model.predict([self.Xc_val, self.Xa_val])
            preds_orig = self.reward_scaler.inverse_transform(preds).flatten()
            targets_orig = self.reward_scaler.inverse_transform(self.y_val.reshape(-1, 1)).flatten()

            print(f"\n--- Debugging epoch {epoch+1} ---")
            for i in range(5):
                print(f"Predicted: {preds_orig[i]:.2f}, True: {targets_orig[i]:.2f}")

            # Optional: scatter plot for visual check
            plt.figure(figsize=(5,5))
            plt.scatter(targets_orig, preds_orig, alpha=0.5)
            plt.xlabel("True Reward")
            plt.ylabel("Predicted Reward")
            plt.title(f"Pred vs True at epoch {epoch+1}")
            plt.plot([min(targets_orig), max(targets_orig)],
                     [min(targets_orig), max(targets_orig)], 'r--')  # diagonal line
            plt.show()
            plt.savefig(f'logs/debugpred_{epoch}.png')

def evaluate_model_bias(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    print(f"MAE: {mae:.4f}, Bias (mean error): {bias:.4f}")
    return mae, bias

# Optional smoothing window (currently unused)
def smooth_series(series, window):
    return pd.Series(series).rolling(window).mean().to_numpy()

# Constraint class to enforce positive weights
class Positive(Constraint):
    def __call__(self, w):
        return tf.abs(w)

# Combines context (r) and action (d) outputs
def _combine_r_and_d(x):
    r, d = x
    return r * (1. - d)

class LSTM_predictor:
    def __init__(self, data_path, save_path='./', window_size=7, lookback_days=21, lstm_size=32):
        self.data_path = data_path
        self.window_size = window_size
        self.lookback_days = lookback_days
        self.lstm_size = lstm_size
        self.model = None
        self.training_model = None
        self.save_path = save_path
        self.iteration = float(data_path.split('_')[-1])
        self.X_context, self.X_action, self.y, self.scenario_ids = self._load_and_process_data()

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        train_idx, val_idx = next(gss.split(self.X_context, groups=self.scenario_ids))

        self.Xc_train, self.Xc_val = self.X_context[train_idx], self.X_context[val_idx]
        self.Xa_train, self.Xa_val = self.X_action[train_idx], self.X_action[val_idx]
        self.y_train, self.y_val = self.y[train_idx], self.y[val_idx]

        self.reward_scaler = StandardScaler()
        self.y_train = self.reward_scaler.fit_transform(self.y_train.reshape(-1, 1)).flatten()
        self.y_val = self.reward_scaler.transform(self.y_val.reshape(-1, 1)).flatten()

    # def _add_rolling_features(self, arr, window=3):
    #     df = pd.DataFrame(arr)
    #     features = [df]

    #     rolling_mean = df.rolling(window=window, min_periods=1).mean()
    #     rolling_std = df.rolling(window=window, min_periods=1).std().fillna(0)

    #     lag_1 = df.shift(1).fillna(0)

    #     features.extend([rolling_mean, rolling_std, lag_1])

    #     augmented = np.concatenate([f.values for f in features], axis=1)

    #     expected_len = len(df)
    #     if augmented.shape[0] != expected_len:
    #         raise ValueError(f"Rolling feature shape mismatch: got {augmented.shape}, expected {expected_len}")

    #     # print(np.array(augmented).shape)
    #     return augmented

    def save_model_and_scaler(self, weights_path='model.weights.h5', scaler_path='reward_scaler.pkl'):
        if self.model is None:
            raise ValueError("Model has not been trained.")

        self.model.save_weights(self.save_path+weights_path)
        print(f"Saved model weights to: {self.save_path+weights_path}")

        if hasattr(self, 'reward_scaler'):
            with open(self.save_path+scaler_path, 'wb') as f:
                pickle.dump(self.reward_scaler, f)
            print(f"Saved reward scaler to: {self.save_path+scaler_path}")
        else:
            print("No reward scaler found. Did you normalize the rewards?")

    def load_model_and_scaler(self, weights_path='model.weights.h5', scaler_path='reward_scaler.pkl'):
        if self.X_context is None or self.X_action is None:
            raise ValueError("Load or prepare data first (needed to build model shape).")

        obs_dim = self.X_context.shape[2]
        act_dim = self.X_action.shape[2]

        self.model, self.training_model = self._construct_model(obs_dim, act_dim)
        self.model.load_weights(self.save_path+weights_path)
        print(f"Loaded model weights from: {self.save_path+weights_path}")

        if os.path.exists(self.save_path+scaler_path):
            with open(self.save_path+scaler_path, 'rb') as f:
                self.reward_scaler = pickle.load(f)
            print(f"Loaded reward scaler from: {self.save_path+scaler_path}")
        else:
            print("Warning: reward scaler file not found.")

    def _load_and_process_data(self):
        all_context = []
        all_action = []
        all_y = []
        all_scenario_ids = []

        csv_files = glob(os.path.join(self.data_path, '*.csv'))

        for file_path in csv_files:
            df = pd.read_csv(file_path, header=None, names=['scenario_id', 'obs', 'action', 'reward'])

            df['obs'] = df['obs'].apply(lambda x: np.array([float(i) for i in str(x).split(';')]))
            df['action'] = df['action'].apply(lambda x: np.array([float(i) for i in str(x).split(';')]))

            scenario_ids = df['scenario_id'].unique()

            for sid in scenario_ids:
                scenario_df = df[df['scenario_id'] == sid].reset_index(drop=True)

                obs_seq = np.stack(scenario_df['obs'].to_numpy())
                act_seq = np.stack(scenario_df['action'].to_numpy())

                rewards = scenario_df['reward'].to_numpy()

                num_steps = len(scenario_df)
                if num_steps <= self.lookback_days:
                    continue

                for t in range(self.lookback_days, num_steps):
                    all_context.append(obs_seq[t - self.lookback_days:t])
                    all_action.append(act_seq[t - self.lookback_days:t])
                    all_y.append(rewards[t])
                    all_scenario_ids.append(sid)

        return (
            np.array(all_context),
            np.array(all_action),
            np.array(all_y, dtype=np.float32),
            np.array(all_scenario_ids)
        )

    def _construct_model(self, obs_dim, act_dim, loss='mse'):
        context_input = Input(shape=(self.lookback_days, obs_dim), name='context_input')
        context_lstm = LSTM(128, name='context_lstm')(context_input)
        
        action_input = Input(shape=(self.lookback_days, act_dim), name='action_input')
        action_lstm = LSTM(128, name='action_lstm')(action_input)

        x = Concatenate()([context_lstm, action_lstm])  # shape (None, 256)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        model_output = Dense(1, activation='linear', name='reward_output')(x)

        model = Model(inputs=[context_input, action_input], outputs=model_output)
        model.compile(loss=loss, optimizer='adam')

        # idk just keep for now 
        training_model = Model(inputs=[context_input, action_input], outputs=model_output)
        training_model.compile(loss=loss, optimizer='adam')

        return model, training_model

    def train(self, epochs=100, batch_size=32, verbose=1):
        if self.X_context is None or self.X_action is None or self.y is None:
            raise ValueError("No training data loaded.")

        obs_dim = self.X_context.shape[2]
        act_dim = self.X_action.shape[2]

        self.model, self.training_model = self._construct_model(obs_dim, act_dim, loss='mse')

        print(self.model.summary())

        # print("X_context.shape:", self.X_context.shape)
        # print("X_action.shape:", self.X_action.shape)
        # print("y.shape:", self.y.shape)
        # print("dtype:", self.X_context.dtype, self.X_action.dtype, self.y.dtype)
        # print(np.isnan(self.X_context).any(), np.isinf(self.X_context).any(),
        # np.isnan(self.X_action).any(), np.isinf(self.X_action).any(),
        # np.isnan(self.y).any(), np.isinf(self.y).any())
        print("Train target stats:", np.min(self.y_train), np.max(self.y_train), np.mean(self.y_train))
        print("Val target stats:", np.min(self.y_val), np.max(self.y_val), np.mean(self.y_val))

        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

        debug_callback = PredictionDebugger(
            model=self.model,
            Xc_val=self.Xc_val,
            Xa_val=self.Xa_val,
            y_val=self.y_val,
            reward_scaler=self.reward_scaler,
            freq=5,
            iteration=self.iteration
        )

        history = self.training_model.fit(
            [self.Xc_train, self.Xa_train], self.y_train,
            validation_data=([self.Xc_val, self.Xa_val], self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, debug_callback],
            verbose=verbose
        )

        self.save_model_and_scaler()
        return history

    def predict(self, context_seq, action_seq):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        context_seq = np.expand_dims(context_seq, axis=0)  # (1, lookback_days, obs_dim)
        action_seq = np.expand_dims(action_seq, axis=0)    # (1, lookback_days, act_dim)
        pred = self.model.predict([context_seq, action_seq], verbose=0)[0][0]
        return self.reward_scaler.inverse_transform([[pred]])[0][0]
    
CONTEXT_COLUMN = 'PredictionRatio'
NB_LOOKBACK_DAYS = 5
NB_TEST_DAYS = 14
WINDOW_SIZE = 10
US_PREFIX = "United States / "
NUM_TRIALS = 1
NUM_EPOCHS = 1000
LSTM_SIZE = 32
MAX_NB_COUNTRIES = 20

if __name__ == "__main__":
    gpu_idx = gpu_id()
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Using GPU {gpu_idx} with memory growth enabled.")
        except RuntimeError as e:
            print(e)
        
    surrogate = LSTM_predictor(
        './data',
        window_size=WINDOW_SIZE,
        lookback_days=NB_LOOKBACK_DAYS,
        lstm_size=LSTM_SIZE
    )

    history = surrogate.train(
        epochs=50,
        batch_size=128,
        verbose=1
    )
    