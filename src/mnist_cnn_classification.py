# %%
# MNIST CNN Classification
import pathlib
import datetime
import numpy as np
import pandas as pd

from tensorflow.keras import callbacks, datasets, layers, models, utils

model_h5 = pathlib.Path("models/mnist_cnn_classification.h5")
hist_pkl = pathlib.Path("models/mnist_cnn_classification_hist.pkl")

# %%
# Prepare Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Add a channels dimension
x_train = np.expand_dims(x_train, axis=3)   # or x_train[..., tf.newaxis]
x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# convert to one-hot vectors
y_train_oh = utils.to_categorical(y_train, 10)
y_test_oh = utils.to_categorical(y_test, 10)

print(x_train.shape, y_train_oh.shape, x_test.shape, y_test_oh.shape)

# %%
# Create Model
if model_h5.exists():
    # load trained model from h5 file
    model = models.load_model(str(model_h5))
else:
    # https://qiita.com/fukuit/items/b3fa460577a0ea139c88
    # https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

model.summary()

# %%
# Training
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# categorical_accuracy will be used internally
# https://stackoverflow.com/questions/43544358

early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

# https://www.tensorflow.org/tensorboard/get_started
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

if hist_pkl.exists():
    hist = pd.read_pickle(str(hist_pkl))
else:
    history = model.fit(x_train, y_train_oh,
                        batch_size=128,
                        epochs=10,
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
test_loss, test_acc = model.evaluate(x_test, y_test_oh, verbose=0)

print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

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
