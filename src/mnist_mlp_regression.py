# %%
# MNIST MLP Regression
import pathlib
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras import callbacks, datasets, layers, models

model_h5 = pathlib.Path("models/mnist_mlp_regression.h5")
hist_pkl = pathlib.Path("models/mnist_mlp_regression_hist.pkl")

# %%
# Prepare Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# %%
# Create Model
if model_h5.exists():
    # load trained model from h5 file
    model = models.load_model(str(model_h5))
else:
    # c.f. https://www.tcom242242.net/entry/2019/03/31/140003/
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

model.summary()

# %%
# Training
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mae', 'mse'])

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

# https://www.tensorflow.org/tensorboard/get_started
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if hist_pkl.exists():
    hist = pd.read_pickle(str(hist_pkl))
else:
    history = model.fit(x_train, y_train,
                        epochs=50,
                        batch_size=32,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=[early_stop, tb_callback])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    model.save(str(model_h5))
    hist.to_pickle(str(hist_pkl))

# %%
# Show Training History
print(hist.tail())
hist.plot(x='epoch', y=['loss', 'val_loss'], grid=True)
hist.plot(x='epoch', y=['mae', 'val_mae'], grid=True)

# start tensorboard
# tensorboard --logdir src/logs/fit
# open http://localhost:6006/

# %%
# Check Model Performance on Test Set
loss, mae, mse = model.evaluate(x_test,  y_test, verbose=2)

print('Test loss:', loss)
print('Test mae:', mae)
print('Test mse:', mse)

# %%
# Predict from Test Set
predictions = model.predict(x_test)
y_pred = predictions.flatten()

predict_df = pd.DataFrame(
    np.stack((y_test, y_pred), axis=1),
    columns=["y_test", "y_pred"]
)

print(predict_df)
predict_df.describe()

# %%
# Evaluate y_pred vs y_test
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# %%
# Show y_pred vs y_test
sns.jointplot(x=y_test, y=y_pred, kind="hex")

error = y_pred - y_test
plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
