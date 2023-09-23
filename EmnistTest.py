# Importing necessary libraries
import numpy as np
import math
import scipy.stats as st
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
# import CHN Layer
from CHNLayer import CHNLayer
from scipy.stats import ttest_rel

emnist_train = tfds.load(name="emnist", split=tfds.Split.TRAIN, batch_size=-1) 
emnist_test = tfds.load(name="emnist", split=tfds.Split.TEST, batch_size=-1)

# Convert to NumPy arrays
emnist_train = tfds.as_numpy(emnist_train) 
emnist_test = tfds.as_numpy(emnist_test)

x_train, y_train = emnist_train["image"], emnist_train["label"] 
x_test, y_test = emnist_test["image"], emnist_test["label"]

# Normalize the images
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Split the training data into training and validation sets (80% training, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
def transpose_images(images):
    return tf.transpose(images, [0, 2, 1, 3])

# Transpose the datasets
x_train = transpose_images(x_train)
x_val = transpose_images(x_val)
x_test = transpose_images(x_test)

##########################################################
mlp_accuracy_history = []
mlp_loss_history = []

chn_loss_history = []
chn_accuracy_history = []

mlp_result_accuracy = []
mlp_result_loss = []

chn_result_accuracy = []
chn_result_loss = []

# declare hyperparameters
epoch = 7
batchSize = 32

MLP_h1 = 768
MLP_h2 = 768
MLP_h3 = 768


CHN_h1 = 768
CHN_h2 = 768
CHN_h3 = 768




optimizer = SGD(lr=0.001, momentum=0.9)
loss = SparseCategoricalCrossentropy(from_logits=True)

#Create MLP model
def create_model_mlp():
    MLP_model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(MLP_h1, activation='relu'),
    Dense(MLP_h2, activation='relu'),
    Dense(MLP_h3, activation='relu'),

    Dense(62 ,activation="softmax")
    ])

    MLP_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    
    return MLP_model

#Create CHN model
def create_model_CHNNet():
    CHN_model = Sequential([
    Flatten(input_shape=(28, 28)),
    CHNLayer(CHN_h1, activation='relu'),
    CHNLayer(CHN_h2, activation='relu'),
    CHNLayer(CHN_h3, activation='relu'),


    Dense(62, activation="softmax")
    ])

    CHN_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['accuracy'])
    return CHN_model
# CHN_model = create_model_CHNNet()
# CHN_model.summary()

num_seeds = 3

# MLP seed
for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    MLP_model = create_model_mlp()
    MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=(x_val, y_val))

    # Evaluate the model on the test data
    test_loss, test_accuracy = MLP_model.evaluate(x_test, y_test)
    mlp_result_accuracy.append(test_accuracy)
    mlp_result_loss.append(test_loss)
    print(f'MLP Loss: {test_loss}\nMLP Accuracy: {test_accuracy}')
    mlp_loss_history.append(MLP_History.history['loss'])
    mlp_accuracy_history.append(MLP_History.history['accuracy'])    
    MLP_model.summary()


# CHN seed
for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    CHN_model = create_model_CHNNet()
    
    CHN_History = CHN_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=(x_val, y_val))

    # Evaluate the model on the test data
    test_loss, test_accuracy = CHN_model.evaluate(x_test, y_test)
    chn_result_accuracy.append(test_accuracy)
    chn_result_loss.append(test_loss)
    print(f'CHN Loss: {test_loss}\nCHN Accuracy: {test_accuracy}')
    chn_loss_history.append(CHN_History.history['loss'])
    chn_accuracy_history.append(CHN_History.history['accuracy'])    
    CHN_model.summary()

# MLP margin
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


# GRAPH

arch = 1
for seed in range(num_seeds):
    plt.plot(mlp_loss_history[seed], label='Training Loss')
    plt.plot(chn_loss_history[seed], label='Training Loss')
    plt.title(f"EMNIST loss: Architecture {arch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend([f"FNN"] + [f"CHN"])
    plt.savefig(f"CHN_MLP_Loss_EMNIST_{seed + 1}_arch_{arch}.pdf")
    plt.show()
# for seed in range(num_seeds):
#     plt.plot(MLP_History.history['val_loss'], label='Validation Loss')
#     plt.plot(CHN_History.history['val_loss'], label='Validation Loss')
#     plt.title("EMNIST Validation")
#     plt.xlabel("Epoch")
#     plt.ylabel("Validation Loss")
#     plt.legend([f"FF"] + [f"CHN"])
#     plt.savefig(f"CHN_MLP_Validation_EMNIST_{seed + 1}.pdf")
#     plt.show()

