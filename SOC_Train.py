import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys

from importlib import reload
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, ReduceLROnPlateau

from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols, CapacityCols
from data_processing.model_data_handler import ModelDataHandler

TIME_STEPS = 8

def create_sequence_data(data_x, data_y):
    seq_data_x = []
    seq_data_y = []

    for i in range(len(data_x) - TIME_STEPS):
        # Ensure the current is not zero at the first and last time steps of the sequence to prevent the sequence from spanning discharge cycles.        
        if data_x[i][0] !=0 and data_x[i+TIME_STEPS-1][0] != 0:
            seq_data_x.append(data_x[i:i+TIME_STEPS])
            seq_data_y.append(data_y[i+TIME_STEPS])

    seq_data_x = np.array(seq_data_x)
    seq_data_y = np.array(seq_data_y)

    return seq_data_x, seq_data_y

data_path = "./"
sys.path.append(data_path)

# Config logging
reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

# Load the cycle and capacity data to memory based on the specified chunk size
dataset = UniboPowertoolsData(
    test_types=['S'],
    chunk_size=1000000,
    lines=[37, 40],
    charge_line=37,
    discharge_line=40,
    base_path=data_path
)

# Prepare the training and testing data for model data handler to load the model input and output data.
train_data_test_names = [
    '000-DM-3.0-4019-S', 
    '001-DM-3.0-4019-S', 
#    '002-DM-3.0-4019-S', 
    '006-EE-2.85-0820-S', 
    '007-EE-2.85-0820-S', 
    '018-DP-2.00-1320-S', 
    '019-DP-2.00-1320-S',
#    '036-DP-2.00-1720-S', 
#    '037-DP-2.00-1720-S', 
#    '038-DP-2.00-2420-S', 
#    '040-DM-4.00-2320-S',
    '042-EE-2.85-0820-S', 
    '045-BE-2.75-2019-S'
]

test_data_test_names = [
    '003-DM-3.0-4019-S',
#    '008-EE-2.85-0820-S',
#    '039-DP-2.00-2420-S', 
#    '041-DM-4.00-2320-S',    
]

dataset.prepare_data(train_data_test_names, test_data_test_names)

# Model data handler will be used to get the model input and output data for further training purpose.
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE,
], [CapacityCols.SOH])

train_x, train_y, test_x, test_y = mdh.get_discharge_whole_cycle(soh = False, output_capacity = False)

train_y = mdh.keep_only_capacity(train_y, is_multiple_output = True)
test_y = mdh.keep_only_capacity(test_y, is_multiple_output = True)

# Flatten train/test dataset to create time sequence dataset
train_x_flat = train_x.reshape(-1, train_x.shape[2])
train_y_flat = train_y.reshape(-1, 1)
test_x_flat = test_x.reshape(-1, test_x.shape[2])
test_y_flat = test_y.reshape(-1, 1)

train_x_seq, train_y_seq = create_sequence_data(train_x_flat, train_y_flat)
test_x_seq, test_y_seq = create_sequence_data(test_x_flat, test_y_flat)

charge_x_scaler, discharge_x_scaler = mdh.get_scalers()
print(charge_x_scaler[0].data_max_)
print(charge_x_scaler[1].data_max_)
print(charge_x_scaler[2].data_max_)
print(charge_x_scaler[3].data_max_)

EXPERIMENT = "lstm_soc_percentage"

experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
print(experiment_name)

opt = tf.keras.optimizers.Adam(learning_rate=0.00003)

# Model implementation
model = Sequential()
model.add(LSTM(256,
                return_sequences=True,
                unroll=True,
                input_shape=(TIME_STEPS, train_x.shape[2])))
model.add(Dropout(0.2)) # 隨機丟棄 20% 神經元，防止過度擬合

model.add(LSTM(256, unroll=True, return_sequences=True))
model.add(Dropout(0.2)) # 隨機丟棄 20% 神經元，防止過度擬合

model.add(LSTM(128, unroll=True, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(data_path + 'results/trained_model/%s_best.keras' % experiment_name, 
                             save_best_only=True, 
                             monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001) # 停滯 5 個 epoch 就把學習率砍半

history = model.fit(train_x_seq, train_y_seq,
                                epochs=5,
                                batch_size=128,
                                verbose=1,
                                validation_split=0.2,
                                callbacks = [es, mc, reduce_lr]
                               )

model.save(data_path + 'results/trained_model/%s.keras' % experiment_name)

hist_df = pd.DataFrame(history.history)
hist_csv_file = data_path + 'results/trained_model/%s_history.csv' % experiment_name
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Load best model
loaded_model = keras.models.load_model(data_path + 'results/trained_model/%s_best.keras' % experiment_name)

# Testing
results = loaded_model.evaluate(test_x_seq, test_y_seq)
print(results)

# Visualiztion
# train loss
fig = go.Figure()
fig.add_trace(go.Scatter(y=history.history['loss'],
                    mode='lines', name='train'))
fig.add_trace(go.Scatter(y=history.history['val_loss'],
                    mode='lines', name='validation'))
fig.update_layout(title='Loss trend',
                  xaxis_title='epoch',
                  yaxis_title='loss',
                  width=1400,
                  height=600)
fig.show()

# train dateset prediction result
train_predictions = loaded_model.predict(train_x_seq)
cycle_num = 0
steps_num = train_x_seq.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=train_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC predicted'))
fig.add_trace(go.Scatter(x=step_index, y=train_y_seq.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC actual'))
fig.update_layout(title='Results on training',
                  xaxis_title='Cycle',
                  yaxis_title='SoC percentage',
                  width=1400,
                  height=600)
fig.show()

# test dateset prediction result
test_predictions = loaded_model.predict(test_x_seq)
cycle_num = 0
steps_num = test_x_seq.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=test_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC predicted'))
fig.add_trace(go.Scatter(x=step_index, y=test_y_seq.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoC actual'))
fig.update_layout(title='Results on testing',
                  xaxis_title='Cycle',
                  yaxis_title='SoC percentage',
                  width=1400,
                  height=600)
fig.show()

# Convert to INT8 tflite model.
def representative_dataset():
    for input_value in tf.data.Dataset.from_tensor_slices((train_x_seq)).batch(1).take(1000):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

# Save the model.
with open('results/trained_model/BMS_SOC_INT8.tflite', 'wb') as f:
  f.write(tflite_quant_model)


def export_numpy_to_c_header(x_array, y_array, filename="test_data.h"):
    """
    將 NumPy 陣列轉換並寫入 C Header 檔案中。
    """
    print(f"正在匯出資料至 {filename} ...")
    
    with open(filename, 'w') as f:
        f.write("#ifndef TEST_DATA_H\n")
        f.write("#define TEST_DATA_H\n\n")

        def write_array_to_c(arr, array_name):
            flat_arr = arr.flatten()
            length = len(flat_arr)
            
            # 寫入陣列維度資訊當作註解，方便 C 語言開發時參考
            f.write(f"// Original array shape: {arr.shape}\n")
            f.write(f"const int {array_name}_dim[] = {{{', '.join(map(str, arr.shape))}}};\n")
            f.write(f"const int {array_name}_length = {length};\n\n")
            
            # 宣告 C 陣列 (這裡以 float 為例)
            f.write(f"const float {array_name}[{length}] = {{\n")
            
            # 將數值分批寫入，避免單行過長導致編譯器報錯 (每行 32 個數值)
            for i in range(0, length, 32):
                chunk = flat_arr[i:i+32]
                chunk_str = ", ".join([f"{val:.6f}" for val in chunk])
                if i + 32 < length:
                    f.write(f"    {chunk_str},\n")
                else:
                    f.write(f"    {chunk_str}\n")
            f.write("};\n\n")

        # 寫入 X 與 Y 資料
        write_array_to_c(x_array, "test_x_seq")
        write_array_to_c(y_array, "test_y_seq")

        f.write("#endif // TEST_DATA_H\n")
    
    print("匯出完成！")

# 使用 sequence 的 test_x_seq 與 test_y_seq 
export_filepath = data_path + 'results/trained_model/SOC_test_data.h'
export_numpy_to_c_header(test_x_seq, test_y_seq, filename=export_filepath)
