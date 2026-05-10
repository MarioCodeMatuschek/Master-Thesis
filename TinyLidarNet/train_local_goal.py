# Variant: LOCAL GOAL only.
# Scalar input = [d_next, a_next] (shape 2) — distance and bearing to the
# next waypoint only.  The final-goal scalars (d_goal, a_goal) are excluded.
import os
import csv
import json
import glob
import math
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # uncomment to force CPU
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

gpu_available = tf.config.list_physical_devices('GPU')
print('GPU AVAILABLE:', bool(gpu_available))

#========================================================
# Functions
#========================================================

def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    return np.mean(loss)

#========================================================
# Global Config
# MAX_DIST is NOT a global constant — room dimensions vary across scenarios.
# It is computed per scenario from scenario_meta.json:
#   spec.width / spec.height  →  max_dist = √(width² + height²)
#========================================================

DATASET_DIR  = '../thesis_dataset'
NUM_RAYS     = 720
MAX_RANGE    = 12.0    # LiDAR maximum range in metres
TRAIN_RATIO  = 0.85
RANDOM_SEED  = 62

model_name        = 'TLN_local'
model_files       = [
    './Models/' + model_name + '_noquantized.tflite',
    './Models/' + model_name + '_int8.tflite',
]
loss_figure_path  = './Figures/loss_curve_local.png'
lr                = 5e-5
loss_function     = 'huber'
batch_size        = 64
num_epochs        = 20
hz                = 40

#========================================================
# Load Dataset
# Scalar features: [d_next, a_next] — local goal only (shape 2 per sample).
# d_goal and a_goal are computed but NOT included in all_scalars.
#========================================================

all_lidar_2ch = []   # will become (N, 720, 2)
all_scalars   = []   # will become (N, 2)  ← local goal only
all_steering  = []   # will become (N,)

for scenario_dir in sorted(glob.glob(os.path.join(DATASET_DIR, 'scenario_*'))):

    meta_path = os.path.join(scenario_dir, 'scenario_meta.json')
    with open(meta_path) as mf:
        meta = json.load(mf)
    w        = meta['spec']['width']
    h        = meta['spec']['height']
    max_dist = math.sqrt(w ** 2 + h ** 2)

    csv_path = os.path.join(scenario_dir, 'scans_long.csv')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    by_scan = {}
    for row in rows:
        by_scan.setdefault(row['scan_id'], []).append(row)

    for sid, group in sorted(by_scan.items()):
        r_m   = np.array([float(r['r_m'])  for r in group], dtype=np.float32)
        valid = np.array([float(r['valid']) for r in group], dtype=np.float32)

        pose_x   = float(group[0]['pose_x'])
        pose_y   = float(group[0]['pose_y'])
        pose_yaw = float(group[0]['pose_yaw'])

        json_path = os.path.join(
            scenario_dir, 'scan_paths', f'scan_{int(sid):04d}.json')
        with open(json_path) as jf:
            path_data = json.load(jf)
        nxt = path_data['path'][1]

        # Steering label
        dx, dy  = nxt['x'] - pose_x, nxt['y'] - pose_y
        bearing = math.atan2(dy, dx)
        delta   = (bearing - pose_yaw + math.pi) % (2 * math.pi) - math.pi

        # LiDAR channels
        d_tilde = np.where(valid == 1, np.clip(r_m, 0.0, MAX_RANGE), 0.0) / MAX_RANGE
        h_k     = valid

        # Local goal scalars only
        dx_n, dy_n = nxt['x'] - pose_x, nxt['y'] - pose_y
        d_next     = math.sqrt(dx_n**2 + dy_n**2) / max_dist
        a_next     = ((math.atan2(dy_n, dx_n) - pose_yaw + math.pi) % (2*math.pi) - math.pi) / math.pi

        all_lidar_2ch.append(np.stack([d_tilde, h_k], axis=-1))  # (720, 2)
        all_scalars.append([d_next, a_next])                       # (2,) — local only
        all_steering.append(delta / math.pi)

# Post-loop assembly
all_lidar_2ch = np.asarray(all_lidar_2ch, dtype=np.float32)  # (N, 720, 2)
all_scalars   = np.asarray(all_scalars,   dtype=np.float32)  # (N, 2)
all_steering  = np.asarray(all_steering,  dtype=np.float32)  # (N,)

all_lidar_2ch, all_scalars, all_steering = shuffle(
    all_lidar_2ch, all_scalars, all_steering, random_state=RANDOM_SEED)

train_n        = int(TRAIN_RATIO * len(all_lidar_2ch))
lidar_2ch      = all_lidar_2ch[:train_n]
scalars        = all_scalars[:train_n]
steering       = all_steering[:train_n, np.newaxis]
test_lidar_2ch = all_lidar_2ch[train_n:]
test_scalars   = all_scalars[train_n:]
test_steering  = all_steering[train_n:, np.newaxis]

print(f'Train samples: {len(lidar_2ch)}, Test samples: {len(test_lidar_2ch)}')
print(f'lidar_2ch: {lidar_2ch.shape}, scalars: {scalars.shape}, steering: {steering.shape}')

#======================================================
# DNN Architecture — local goal variant
# scalar_in shape is (2,): [d_next, a_next]
#======================================================

lidar_in  = tf.keras.Input(shape=(NUM_RAYS, 2), name='lidar')
scalar_in = tf.keras.Input(shape=(2,),          name='scalars_local')

x = tf.keras.layers.Conv1D(24, 10, strides=4, activation='relu')(lidar_in)
x = tf.keras.layers.Conv1D(36,  8, strides=4, activation='relu')(x)
x = tf.keras.layers.Conv1D(48,  4, strides=2, activation='relu')(x)
x = tf.keras.layers.Conv1D(64,  3,            activation='relu')(x)
x = tf.keras.layers.Conv1D(64,  3,            activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Concatenate()([x, scalar_in])
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(50,  activation='relu')(x)
x = tf.keras.layers.Dense(10,  activation='relu')(x)
out = tf.keras.layers.Dense(1, activation='tanh', name='steering')(x)

model = tf.keras.Model(inputs=[lidar_in, scalar_in], outputs=out)

#======================================================
# Model Compilation
#======================================================

optimizer = Adam(lr)
model.compile(optimizer=optimizer, loss=loss_function)
print(model.summary())

#======================================================
# Model Fit
#======================================================

start_time = time.time()
history = model.fit(
    [lidar_2ch, scalars],
    steering,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=([test_lidar_2ch, test_scalars], test_steering)
)
print(f'=============>{int(time.time() - start_time)} seconds<=============')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (local goal)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(loss_figure_path)
plt.close()

#======================================================
# Model Evaluation
#======================================================

print("==========================================")
print("Model Evaluation")
print("==========================================")

test_loss = model.evaluate([test_lidar_2ch, test_scalars], test_steering)
print(f'Overall Test Loss = {test_loss}')

y_pred = model.predict([test_lidar_2ch, test_scalars])
hl = huber_loss(test_steering, y_pred)
print(f'\nOverall Huber Loss: {hl:.3f}')

steering_test_loss = huber_loss(test_steering, y_pred)
print(f'Steering Test Loss: {steering_test_loss:.4f}')

#======================================================
# Save Model — TFLite export
#======================================================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_model_path = './Models/' + model_name + '_noquantized.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
    print(f'{model_name}_noquantized.tflite is saved.')

def representative_data_gen():
    for i in range(len(lidar_2ch)):
        yield [
            lidar_2ch[i:i+1].astype(np.float32),
            scalars[i:i+1].astype(np.float32),
        ]

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
quantized_tflite_model = converter.convert()

tflite_model_path = './Models/' + model_name + '_int8.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(quantized_tflite_model)
    print(f'{model_name}_int8.tflite is saved.')

print('TFLite models saved.')

#======================================================
# Evaluate TFLite Models
#======================================================

def evaluate_model(model_path, test_lidar_2ch, test_scalars, test_steering):
    """Evaluate a TFLite model on the test set; return predictions and inference times."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details    = interpreter.get_input_details()
    lidar_input_idx  = input_details[0]['index']
    scalar_input_idx = input_details[1]['index']
    output_details   = interpreter.get_output_details()

    period = 1.0 / hz
    inference_times_micros = []
    output_steering = []

    for i, lidar_sample in enumerate(test_lidar_2ch):
        l_in = lidar_sample[np.newaxis].astype(np.float32)   # (1, 720, 2)
        s_in = test_scalars[i:i+1].astype(np.float32)         # (1, 2)

        ts = time.time()
        interpreter.set_tensor(lidar_input_idx,  l_in)
        interpreter.set_tensor(scalar_input_idx, s_in)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        dur = time.time() - ts

        inference_times_micros.append(dur * 1e6)
        if dur > period:
            print('%.3f: took %.2f microseconds - deadline miss.' % (dur, int(dur * 1e6)))
        output_steering.append(output[0, 0])

    y_pred = np.asarray(output_steering)[:, np.newaxis]

    arr    = np.array(inference_times_micros)
    perc99 = np.percentile(arr, 99)
    arr    = arr[arr < perc99]
    print('Model: ', model_path)
    print('Average Inference Time: %.2f microseconds' % np.mean(arr))
    print('Maximum Inference Time: %.2f microseconds' % np.max(arr))

    return y_pred, inference_times_micros


all_inference_times_micros = []
for model_file in model_files:
    y_pred, inference_times_micros = evaluate_model(
        model_file, test_lidar_2ch, test_scalars, test_steering)
    all_inference_times_micros.append(inference_times_micros)
    print(f'Huber Loss for {model_file}: {huber_loss(test_steering, y_pred):.4f}\n')

plt.figure()
for inference_times_micros in all_inference_times_micros:
    arr    = np.array(inference_times_micros)
    perc99 = np.percentile(arr, 99)
    arr    = arr[arr < perc99]
    plt.plot(arr)
plt.xlabel('Inference Iteration')
plt.ylabel('Inference Time (microseconds)')
plt.title('Inference Time per Iteration (local goal)')
plt.legend(model_files)

print('End')
