# %%
"""baseline model"""
# MNIST MLP Classification
# the code here is mostly from Tensorflow official document
# https://www.tensorflow.org/tutorials/quickstart/beginner
# https://www.tensorflow.org/tutorials/keras/classification

import pathlib
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import callbacks, datasets, layers, losses, models

model_h5 = pathlib.Path("models/mnist_mlp_classification.h5")
hist_pkl = pathlib.Path("models/mnist_mlp_classification_hist.pkl")

# %%
# Prepare Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

plt.figure()
plt.imshow(x_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# %%
# Create Model
if model_h5.exists():
    # load trained model from h5 file
    model = models.load_model(str(model_h5))
else:
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10)
    ])

model.summary()

# %%
# Show Model Outputs Without Training
# logits (raw (non-normalized) predictions)
predictions = model(x_train[:1]).numpy()
print(predictions)

# convert logits to probabilities
probabilities = tf.nn.softmax(predictions).numpy()
print(probabilities)

# This untrained model gives probabilities close to random (1/10 for each class)
# so the initial loss should be close to -tf.math.log(1/10) ~= 2.3
loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
print(loss_fn(y_train[:1], predictions).numpy())

# %%
# Training
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# above is same as loss="sparse_categorical_crossentropy"

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
hist.plot(x='epoch', y=['accuracy', 'val_accuracy'], grid=True)

# start tensorboard
# tensorboard --logdir src/logs/fit
# open http://localhost:6006/

# %%
# Check Model Performance on Test Set
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# %%
# Show Model Outputs from Test Set
probability_model = models.Sequential([
    model,
    layers.Softmax()
])

# show first 5 sample data
probs = (probability_model(x_test[:5]))
print(np.round(probs, 2))
print(y_test[:5])

# %%
# Predict from Test Set
predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)

predict_df = pd.DataFrame(
    np.stack((y_test, y_pred), axis=1),
    columns=["y_test", "y_pred"]
)

print(predict_df)
predict_df.pivot_table(index='y_test', columns='y_pred', aggfunc='size')
