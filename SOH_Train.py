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
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
#from keras.utils import np_utils
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Masking
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback

data_path = "./"
sys.path.append(data_path)

from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols, CapacityCols
from data_processing.model_data_handler import ModelDataHandler

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
    '002-DM-3.0-4019-S', 
    '006-EE-2.85-0820-S', 
    '007-EE-2.85-0820-S', 
    '018-DP-2.00-1320-S', 
    '019-DP-2.00-1320-S',
    '036-DP-2.00-1720-S', 
    '037-DP-2.00-1720-S', 
    '038-DP-2.00-2420-S', 
    '040-DM-4.00-2320-S',
    '042-EE-2.85-0820-S', 
    '045-BE-2.75-2019-S'
]

test_data_test_names = [
    '003-DM-3.0-4019-S',
    '008-EE-2.85-0820-S',
    '039-DP-2.00-2420-S', 
    '041-DM-4.00-2320-S',    
]

dataset.prepare_data(train_data_test_names, test_data_test_names)

# Model data handler will be used to get the model input and output data for further training purpose.
mdh = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE,
])

train_x, train_y, test_x, test_y = mdh.get_charge_whole_cycle(soh = True, output_capacity = False, multiple_output=True)

train_y = mdh.keep_only_capacity(train_y, is_multiple_output = True)
test_y = mdh.keep_only_capacity(test_y, is_multiple_output = True)

train_y = train_y[:, [0]]
test_y = test_y[:, [0]]

print(f"Change train_y shape to {train_y.shape}")
print(f"Change test_y shape to {test_y.shape}")

#print(train_x)

# Min-Max Scaler is a popular data normalization
# Xscaled = (X - Xmin) / (Xmax - Xmin)
charge_x_scaler, discharge_x_scaler = mdh.get_scalers()
print(charge_x_scaler[0].data_max_)   
print(charge_x_scaler[0].data_min_)   
print(charge_x_scaler[1].data_max_)
print(charge_x_scaler[1].data_min_)   
print(charge_x_scaler[2].data_max_)
print(charge_x_scaler[2].data_min_)   

EXPERIMENT = "cnn_soh_percentage"

experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
print(experiment_name)

opt = tf.keras.optimizers.Adam(learning_rate=0.00003)

# Model implementation
input = keras.Input(shape=(train_x.shape[1], train_x.shape[2]))

x = layers.Conv1D(64, 32, activation='relu')(input)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Conv1D(64, 32, activation='relu')(x)
x = layers.MaxPooling1D(pool_size = 2)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
output = layers.Dense(1, activation='relu')(x)

model = keras.Model(inputs=input, outputs=output)
model.summary()

# Model compile
model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Setup early stop and check point
es = EarlyStopping(monitor='val_loss', patience=50)
mc = ModelCheckpoint(data_path + 'results/trained_model/%s_best.keras' % experiment_name, 
                             save_best_only=True,
                             monitor='val_loss')

history = model.fit(train_x, train_y,
                                epochs=30,
                                batch_size=32,
                                verbose=1,
                                validation_split=0.2,
                                callbacks = [es, mc]
                               )

model.save(data_path + 'results/trained_model/%s.keras' % experiment_name)

hist_df = pd.DataFrame(history.history)
hist_csv_file = data_path + 'results/trained_model/%s_history.csv' % experiment_name
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# Load best model
loaded_model = keras.models.load_model(data_path + 'results/trained_model/%s_best.keras' % experiment_name)

# Testing
results = loaded_model.evaluate(test_x, test_y)
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
train_predictions = loaded_model.predict(train_x)
cycle_num = 0
steps_num = train_x.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=train_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH predicted'))
fig.add_trace(go.Scatter(x=step_index, y=train_y.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH actual'))
fig.update_layout(title='Results on training',
                  xaxis_title='Cycle',
                  yaxis_title='SoH percentage',
                  width=1400,
                  height=600)
fig.show()

# test dateset prediction result
test_predictions = loaded_model.predict(test_x)
cycle_num = 0
steps_num = test_x.shape[0]
step_index = np.arange(cycle_num*steps_num, (cycle_num+1)*steps_num)

fig = go.Figure()
fig.add_trace(go.Scatter(x=step_index, y=test_predictions.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH predicted'))
fig.add_trace(go.Scatter(x=step_index, y=test_y.flatten()[cycle_num*steps_num:(cycle_num+1)*steps_num],
                    mode='lines', name='SoH actual'))
fig.update_layout(title='Results on testing',
                  xaxis_title='Cycle',
                  yaxis_title='SoH percentage',
                  width=1400,
                  height=600)
fig.show()

# Convert to INT8 tflite model.
def representative_dataset():
    for input_value in tf.data.Dataset.from_tensor_slices((train_x)).batch(1).take(1000):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()

# Save the model.
with open('results/trained_model/BMS_SOH_INT8.tflite', 'wb') as f:
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
            
            # 將數值分批寫入，避免單行過長導致編譯器報錯 (每行 10 個數值)
            for i in range(0, length, 10):
                chunk = flat_arr[i:i+10]
                chunk_str = ", ".join([f"{val:.6f}" for val in chunk])
                if i + 10 < length:
                    f.write(f"    {chunk_str},\n")
                else:
                    f.write(f"    {chunk_str}\n")
            f.write("};\n\n")

        # 寫入 X 與 Y 資料
        write_array_to_c(x_array, "test_x")
        write_array_to_c(y_array, "test_y")

        f.write("#endif // TEST_DATA_H\n")
    
    print("匯出完成！")

# Export test_x and test_y data to C header file for later use in C language development.
export_filepath = data_path + 'results/trained_model/SOH_test_data.h'
export_numpy_to_c_header(test_x, test_y, filename=export_filepath)
