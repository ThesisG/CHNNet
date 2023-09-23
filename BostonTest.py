# import necessary libraries & modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD

# import CHN Layer
from CHNLayer import CHNLayer
from scipy.stats import ttest_rel
import math

# load dataset & do slight pre-processing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(
    path='boston_housing.npz', test_split=0.2, seed=113
)


mms = MinMaxScaler() 
mms.fit(x_train)
x_train = mms.transform(x_train)
x_test = mms.transform(x_test)

#################################################################
mlp_loss_history = []

chn_loss_history = []
# declare hyperparameters
epoch = 50
batchSize = 128

MLP_h1 = 160
MLP_h2 = 128
MLP_h3 = 96
MLP_h4 = 64

CHN_h1 = 160
CHN_h2 = 128
CHN_h3 = 96
CHN_h4 = 64








optimizer = RMSprop(learning_rate=0.0002)
loss = 'mse'

mlp_result = []
def create_model_mlp():
    MLP_model = Sequential([
    Dense(MLP_h1, input_dim = 13, activation="relu"),
    Dense(MLP_h2, activation="relu"),
    Dense(MLP_h3, activation="relu"),
    Dense(MLP_h4, activation="relu"),
    # Dense(MLP_h5, activation="relu"),
    # Dense(MLP_h6, activation="relu"),

    Dense(1, activation="linear")
    ])

    MLP_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['mae'])
    
    return MLP_model


chn_result = []
def create_model_CHNNet():
    CHN_model = Sequential([
    CHNLayer(CHN_h1, input_dim = 13, activation="relu"),
    CHNLayer(CHN_h2, activation="relu"),
    CHNLayer(CHN_h3, activation="relu"),
    CHNLayer(CHN_h4, activation="relu"),
    # CHNLayer(CHN_h5, activation="relu"),
    # CHNLayer(CHN_h6, activation="relu"),

    Dense(1, activation="linear"),
    ])

    CHN_model.compile(optimizer=optimizer,
                loss=loss,
                metrics=['mae'])
    return CHN_model

num_seeds = 3

for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    MLP_model = create_model_mlp()
    MLP_History = MLP_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)
    # Evaluate the model on the test data
    test_loss, test_accuracy = MLP_model.evaluate(x_test, y_test)
    mlp_result.append(test_loss)
    mlp_loss_history.append(MLP_History.history['loss'])
    MLP_model.summary()

for seed in range(num_seeds):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    CHN_model = create_model_CHNNet()
    
    CHN_History = CHN_model.fit(x_train, y_train, epochs = epoch, batch_size = batchSize, validation_data=None)

    # Evaluate the model on the test data
    test_loss, test_accuracy = CHN_model.evaluate(x_test, y_test)
    chn_result.append(test_loss)
    print(f'CHN Loss: {test_loss}')
    chn_loss_history.append(CHN_History.history['loss'])
    CHN_model.summary()


def mlp_stats():
  
  mlp_loss_mean = np.mean(mlp_result)
  mlp_loss_std = np.std(mlp_result)
  mlp_loss_var = np.var(mlp_result)
  mlp_margin_loss_error = 1.96 * (mlp_loss_std / math.sqrt(len(mlp_result)))
  mlp_confidence_loss_interval = (mlp_loss_mean - mlp_margin_loss_error, mlp_loss_mean + mlp_margin_loss_error)

  print(f"MLP loss mean: {mlp_loss_mean}")
  print(f"MLP loss std: {mlp_loss_std}")
  print(f"MLP loss var: {mlp_loss_var}")
  print(f"MLP loss error: {mlp_margin_loss_error}")
  print(f"MLP confidence loss interval: {mlp_confidence_loss_interval}")


def chn_stats():
  chn_loss_mean = np.mean(chn_result)
  chn_loss_std = np.std(chn_result)
  chn_loss_var = np.var(chn_result)
  chn_margin_loss_error = 1.96 * (chn_loss_std / math.sqrt(len(chn_result)))
  chn_confidence_loss_interval = (chn_loss_mean - chn_margin_loss_error, chn_loss_mean + chn_margin_loss_error)

  print(f"CHN loss mean: {chn_loss_mean}")
  print(f"CHN loss std: {chn_loss_std}")
  print(f"CHN loss var: {chn_loss_var}")
  print(f"CHN loss error: {chn_margin_loss_error}")
  print(f"CHN confidence loss interval: {chn_confidence_loss_interval}")

# t and p val
mlp_stats()
chn_stats()

t_statistic_loss, p_value_loss = ttest_rel(mlp_result, chn_result)
print("t-statistic:", t_statistic_loss)
print("p-value:", p_value_loss)

arch = 3
for seed in range(num_seeds):
    plt.plot(mlp_loss_history[seed], label='Training Loss')
    plt.plot(chn_loss_history[seed], label='Training Loss')
    plt.title(f"Boston loss: Architecture {arch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend([f"FNN"] + [f"CHN"])
    plt.savefig(f"CHN_MLP_Loss_Boston_{seed + 1}_arch_{arch}.pdf")
    plt.show()