# import necessary libraries & modules
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np
import math
from scipy.stats import ttest_rel
from tensorflow.keras.optimizers import RMSprop

import tensorflow as tf

# import CHN Layer
from CHNLayer import CHNLayer


# load dataset & do slight pre-processing
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
# declare hyperparameters
epoch = 10
batchSize = 512

MLP_h1 = 360
MLP_h2 = 334
MLP_h3 = 304
MLP_h4 = 268
MLP_h5 = 238
MLP_h6 = 208
MLP_h7 = 176
MLP_h8 = 142


CHN_h1 = 288
CHN_h2 = 256
CHN_h3 = 224
CHN_h4 = 192
CHN_h5 = 160
CHN_h6 = 128
CHN_h7 = 96
CHN_h8 = 64

optimizer = RMSprop(learning_rate=0.0001)
loss = SparseCategoricalCrossentropy(from_logits=True)

# declare hyperparameters
mlp_result_accuracy = []
mlp_result_loss = []
mlp_loss_history = []

def create_model_mlp():
    MLP_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(MLP_h1, activation='relu'),
    Dense(MLP_h2, activation='relu'),
    Dense(MLP_h3, activation='relu'),
    Dense(MLP_h4, activation='relu'),
    Dense(MLP_h5, activation='relu'),
    Dense(MLP_h6, activation='relu'),
    Dense(MLP_h7, activation='relu'),
    Dense(MLP_h8, activation='relu'),
    # Dense(MLP_h9, activation='relu'),
    # Dense(MLP_h10, activation='relu'),
    Dense(10, activation="softmax")
    ])

    MLP_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    
    return MLP_model


chn_result_accuracy = []
chn_result_loss = []
chn_loss_history = []

def create_model_CHNNet():
    CHN_model = Sequential([
    Flatten(input_shape=(28, 28)),   
    CHNLayer(CHN_h1, activation='relu'),    
    CHNLayer(CHN_h2, activation='relu'),
    CHNLayer(CHN_h3, activation='relu'),
    CHNLayer(CHN_h4, activation='relu'),
    CHNLayer(CHN_h5, activation='relu'),
    CHNLayer(CHN_h6, activation='relu'),
    CHNLayer(CHN_h7, activation='relu'),
    CHNLayer(CHN_h8, activation='relu'),
    # CHNLayer(CHN_h9, activation='relu'),
    # CHNLayer(CHN_h10, activation='relu'),
    Dense(10, activation="softmax")
    ])

    CHN_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    return CHN_model

num_seeds = 3

for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    MLP_model = create_model_mlp()
    MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)
    # Evaluate the model on the test data
    test_loss, test_accuracy = MLP_model.evaluate(x_test, y_test)
    mlp_result_accuracy.append(test_accuracy)
    mlp_result_loss.append(test_loss)
    mlp_loss_history.append(MLP_History.history['loss'])
    MLP_model.summary()

for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    CHN_model = create_model_CHNNet()
    
    CHN_History = CHN_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)

    # Evaluate the model on the test data
    test_loss, test_accuracy = CHN_model.evaluate(x_test, y_test)
    chn_result_accuracy.append(test_accuracy)
    chn_result_loss.append(test_loss)    
    chn_loss_history.append(CHN_History.history['loss'])
    CHN_model.summary()


def mlp_stats():

  mlp_accuracy_mean = np.mean(mlp_result_accuracy)
  mlp_accuracy_std = np.std(mlp_result_accuracy)
  mlp_accuracy_var = np.var(mlp_result_accuracy)
  mlp_margin_accuracy_error = 1.96 * (mlp_accuracy_std / math.sqrt(len(mlp_result_accuracy)))
  mlp_confidence_accuracy_interval = (mlp_accuracy_mean - mlp_margin_accuracy_error, mlp_accuracy_mean + mlp_margin_accuracy_error)

  print(f"MLP accuracy mean: {mlp_accuracy_mean}")
  print(f"MLP accuracy std: {mlp_accuracy_std}")
  print(f"MLP accuracy var: {mlp_accuracy_var}")
  print(f"MLP accuracy error: {mlp_margin_accuracy_error}")
  print(f"MLP confidence accuracy interval: {mlp_confidence_accuracy_interval}")

  
  mlp_loss_mean = np.mean(mlp_result_loss)
  mlp_loss_std = np.std(mlp_result_loss)
  mlp_loss_var = np.var(mlp_result_loss)
  mlp_margin_loss_error = 1.96 * (mlp_loss_std / math.sqrt(len(mlp_result_loss)))
  mlp_confidence_loss_interval = (mlp_loss_mean - mlp_margin_loss_error, mlp_loss_mean + mlp_margin_loss_error)

  print(f"MLP loss mean: {mlp_loss_mean}")
  print(f"MLP loss std: {mlp_loss_std}")
  print(f"MLP loss var: {mlp_loss_var}")
  print(f"MLP loss error: {mlp_margin_loss_error}")
  print(f"MLP confidence loss interval: {mlp_confidence_loss_interval}")

def chn_stats():
  chn_accuracy_mean = np.mean(chn_result_accuracy)
  chn_accuracy_std = np.std(chn_result_accuracy)
  chn_accuracy_var = np.var(chn_result_accuracy)
  chn_margin_accuracy_error = 1.96 * (chn_accuracy_std / math.sqrt(len(chn_result_accuracy)))
  chn_confidence_accuracy_interval = (chn_accuracy_mean - chn_margin_accuracy_error, chn_accuracy_mean + chn_margin_accuracy_error)

  print(f"CHN accuracy mean: {chn_accuracy_mean}")
  print(f"CHN accuracy std: {chn_accuracy_std}")
  print(f"CHN accuracy var: {chn_accuracy_var}")
  print(f"CHN accuracy error: {chn_margin_accuracy_error}")
  print(f"CHN confidence accuracy interval: {chn_confidence_accuracy_interval}")


  chn_loss_mean = np.mean(chn_result_loss)
  chn_loss_std = np.std(chn_result_loss)
  chn_loss_var = np.var(chn_result_loss)
  chn_margin_loss_error = 1.96 * (chn_loss_std / math.sqrt(len(chn_result_loss)))
  chn_confidence_loss_interval = (chn_loss_mean - chn_margin_loss_error, chn_loss_mean + chn_margin_loss_error)

  print(f"CHN loss mean: {chn_loss_mean}")
  print(f"CHN loss std: {chn_loss_std}")
  print(f"CHN loss var: {chn_loss_var}")
  print(f"CHN loss error: {chn_margin_loss_error}")
  print(f"CHN confidence loss interval: {chn_confidence_loss_interval}")

# t and p val
mlp_stats()
chn_stats()

t_statistic_loss, p_value_loss = ttest_rel(mlp_result_loss, chn_result_loss)
print("t-statistic_loss:", t_statistic_loss)
print("p-value_loss:", p_value_loss)
t_statistic_accuracy, p_value_accuracy = ttest_rel(mlp_result_accuracy, chn_result_accuracy)
print("t-statistic_accuracy:", t_statistic_accuracy)
print("p-value_accuracy:", p_value_accuracy)


for seed in range(num_seeds):
    plt.plot(mlp_loss_history[seed], label='Training Loss')
    plt.plot(chn_loss_history[seed], label='Training Loss')
    plt.title("MNIST loss: Architecture 6")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend([f"FF"] + [f"CHN"])
    plt.savefig(f"CHN_MLP_Loss_MNIST_{seed + 1}.pdf")
    plt.show()