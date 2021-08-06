# TensorFlow Image Regression

This is TensorFlow examples for Image Regression as well as usual Classification.

### computer vision model combinations

* MLP Classification
* CNN Classification
* MLP Regression
* CNN Regression

(MLP = Multilayer Perceptron, CNN = Convolutional Neural Network)

### application example

Icon image regression (in `icon-image-regression` folder)

* explore icon image data
* MLP Regression (baseline model)
* CNN Regression (improved model)

## Setup

```bash
# install from Pipfile.lock
pipenv sync
pipenv shell

# use VSCode
code .
```

### MNIST Results Summary

| evaluate on Test set    | MLP           | CNN           |
| -------------           | ------------- | ------------- |
| classification accuracy | 0.9795        | 0.9907        |
| regression R2           | 0.9589        | 0.9530        |

NOTE: Classification should be used for MNIST (Regression is for reference)

## Models Types

* MLP vs CNN
  - MLP = Multilayer Perceptron (classical neural network)
  - CNN = Convolutional Neural Network (current computer vision algorithms)
* Classification vs Regression
  - Classification = Categorical Prediction (predicting a label)
  - Regression = Numeric Prediction (predicting a quantity)

| model type             | Classification   | Regression                  |
| -------------          | -------------    | -------------               |
| prediction             | categorical      | numeric                     |
| last layer activation  | softmax          | linear (i.e. no activation) |
| model output           | 10 probabilities | 1 number                    |
| loss                   | CCE or SCCE      | MSE (best practice)         |
| metrics                | accuracy         | MAE (or MSE, RMSE, R2)      |

## CCE vs SCCE

|           | CCE                        | SCCE                              |
| --------- | -------------              | -------------                     |
| long name | `categorical_crossentropy` | `sparse_categorical_crossentropy` |
| target    | one-hot array              | integers (category indices)       |
| accuracy  | `categorical_accuracy`     | `sparse_categorical_accuracy`     |
| note      | conventional way           | we can skip one hot encoding step |

* `accuracy`: each `xxx_accuracy` will be used internally (auto select)
  - https://stackoverflow.com/questions/43544358
* both will calculate the probabilities and essentially same thing
  - https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/
* about CCE and SCCE
  - https://stackoverflow.com/questions/58565394
  - https://stackoverflow.com/questions/44477489

## Reference

* Tensorflow official docs
  - https://www.tensorflow.org/tutorials/quickstart/beginner
  - https://www.tensorflow.org/tutorials/images/cnn
  - https://www.tensorflow.org/tensorboard/get_started
* VSCode `# %%`
  - https://qiita.com/386jp/items/f023de9457c99b964a85
