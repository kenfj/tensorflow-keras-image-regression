# %%
"""Image Regression Baseline"""
from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from PIL import Image

import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, models

# %%
print(tf.__version__)

# %% [markdown]
# ## Data

# %%
# select distinct
df2 = pd.read_csv('./data/all_sb.csv')
df2 = df2.drop_duplicates()

# select group by id having max rating_count
max_rating_count_ids = df2.groupby(
    ["id"], sort=False, as_index=False
)["rating_count"].max()
merged = df2.merge(max_rating_count_ids, on=["id", "rating_count"])
df3 = merged.drop_duplicates(subset='id', keep="last")
df3.shape

# %%
# calculate weighted average rating
df3.rating_count_list = df3.rating_count_list.apply(json.loads)
df3["rating_value2"] = df3.rating_count_list.apply(
    lambda x:
    (x[0] + x[1]*2 + x[2]*3 + x[3]*4 + x[4]*5) /
    (sum(x) - sys.float_info.epsilon)
)

# %%
df3.tail()

# %%
# create new dataframe with average rating
df = df3[["id", "rating_value2"]].copy()
df.reset_index(drop=True, inplace=True)
df.rename({"rating_value2": "rating"}, axis=1, inplace=True)

df.tail()

# %%


def file_exist(app_id):
    """check if icon file exist"""
    filename = f"./data/icondata/all/{app_id}.png"
    if os.path.exists(filename):
        return True
    filename2 = f"./data/icondata/all/{app_id}.jpg"
    if os.path.exists(filename2):
        return True
    return False


df2 = df[df.id.map(file_exist)]
df2.reset_index(drop=True, inplace=True)
df2.tail()

# %%
df2.describe()

# %%
# # z-score normalization

# # Degrees of freedom
# # default is 0 in scipy.stats.zscore and 1 in pandas.DataFrame.std
# ddof = 0

# # https://stackoverflow.com/questions/51471672/reverse-z-score-pandas-dataframe
# rating_mean = df2.rating.mean()
# rating_std = df2.rating.std(ddof=ddof)

# def reverse_zscore(pandas_series, mean, std):
#     '''Mean and standard deviation should be of original variable before standardization'''
#     yis = pandas_series * std + mean
#     return yis

# from scipy.stats import zscore

# df2["rating_original"] = df2.rating
# df2["rating"] = zscore(df2.rating, ddof=ddof)
# df2.head()

# %%


def img2array(filename):
    """convert image file to numpy array"""
    icon = Image.open(filename)
    img_array = np.array(icon.convert('RGB'), 'f')

    if img_array.shape != (246, 246, 3):
        print(filename)

    img_array /= 255
    return img_array


def get_img_array(app_id):
    """convert id to image array"""
    filename = f"./data/icondata/all/{app_id}.png"
    if os.path.exists(filename):
        return img2array(filename)
    filename2 = f"./data/icondata/all/{app_id}.jpg"
    if os.path.exists(filename2):
        return img2array(filename2)
    raise FileNotFoundError(f"img file not found: {app_id}")


try:
    img = pd.read_pickle("img.pkl")
except FileNotFoundError as err:
    img = df2.id.map(get_img_array)
    img.to_pickle("img.pkl")

# need to copy to avoid SettingWithCopyWarning
# https://stackoverflow.com/a/49603010
df2 = df2.copy()

df2["img"] = img
df2.reset_index(drop=True, inplace=True)

df2.tail()

# %%

X = np.stack(df2.img)   # shape: (***, 246, 246, 3)
Y = df2.rating          # shape: (***,)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# %%
plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# %% [markdown]
# ## Icon Regression Model

# %%
input_shape = x_train[0].shape

model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='linear'),
])

# Huber loss instead of mse for regression
# https://support.dl.sony.com/docs-ja/チュートリアル：入力画像を元に連続値を推定す
model.compile(
    optimizer="rmsprop",
    loss=losses.Huber(),
    metrics=['mae', 'mse']
)

model.summary()

# %% [markdown]
# ## Training

# %%
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

# https://www.tensorflow.org/tensorboard/get_started
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

try:
    model = models.load_model('models/icon_regression_model.h5')
    hist = pd.read_pickle("models/icon_regression_model_hist.pkl")
except OSError as err:
    print(err)
    history = model.fit(x_train, y_train,
                        batch_size=2,
                        epochs=50,
                        verbose=1,
                        validation_split=0.2,
                        callbacks=[early_stop, tb_callback])
    model.save('models/icon_regression_model.h5')

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_pickle("models/icon_regression_model_hist.pkl")

# %% [markdown]
# ## start Tensorboard
# ```
# cd src/
# tensorboard --logdir logs/fit
# open http://localhost:6006/
# ```

# %%
hist.tail()

# %%
hist.plot(x='epoch', y=['mae', 'val_mae'], grid=True)

# %%
hist.plot(x='epoch', y=['mse', 'val_mse'], grid=True)

# %%
hist.plot(x='epoch', y=['loss', 'val_loss'], grid=True)

# %%
loss, mae, mse = model.evaluate(x_test,  y_test, verbose=2)

print('Test loss:', loss)
print('Test mae:', mae)
print('Test mse:', mse)

# %%
predictions = model.predict(x_test)
y_pred = predictions.flatten()

predict_df = pd.DataFrame(
    np.stack((y_test, y_pred), axis=1),
    columns=["y_test", "y_pred"]
)

predict_df

# %%
# note: y_pred can be out of the y_test range
predict_df.describe()

# %%
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")

# %%
sns.jointplot(x=y_test, y=y_pred, kind="hex")

# %%
error = y_pred - y_test
plt.hist(error, bins=25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")

# %%
