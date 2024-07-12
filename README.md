# PyTorch Tutorial Assets

<!-- Badges -->

![Python](https://img.shields.io/badge/Python-3.10.13-grey?logo=python&labelColor=black&style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-grey?logo=pytorch&labelColor=black&style=flat)
![NumPy](https://img.shields.io/badge/NumPy-1.26.4-grey?logo=numpy&labelColor=black&style=flat)

<!-- Introduction -->

This repository contains all the notebooks, models, scripts, and other assets I have created while following the [PyTorch Beginner Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN) by Brian Heintz. It catalogs all the work I have done to learn and implement neural networks in PyTorch, serving as a record of my progress and a resource for gauging my development in the area.

## Repository Assets

### Notebooks

The `notebooks/` directory stores various notebooks referenced from the tutorial. It contains the following:

<!-- Notebooks Table -->

<table>
  <thead>
    <tr>
      <th align=center>S. No.</th>
      <th align=center>Notebook</th>
      <th align=center>Description</th>
      <th align=center>Key Concepts Covered</th>
      <th align=center>Dependencies</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align=center>1.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/01_introduction_to_pytorch.ipynb>01_introduction_to_pytorch.ipynb</a></code></td>
      <td>Introduction to basic PyTorch concepts and training a LeNet-5 model on the <a href=https://paperswithcode.com/dataset/cifar-10><code>CIFAR10</code></a> dataset</td>
      <td>Tensors, Autograd, Models, Datasets, Data Loaders, CNN</td>
      <td><code>matplotlib</code>, <code>numpy</code>, <code>torch</code>, <code>torchvision</code></td>
    </tr>
    <tr>
      <td align=center>2.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/02_introduction_to_pytorch_tensors.ipynb>02_introduction_to_pytorch_tensors.ipynb</a></code></td>
      <td>Introduction to basic PyTorch tensor operations</td>
      <td>Tensors - Creation and Operations, Tensor Manipulation</td>
      <td><code>numpy</code>, <code>torch</code></td>
    </tr>
    <tr>
      <td align=center>3.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/03_the_fundamentals_of_autograd.ipynb>03_the_fundamentals_of_autograd.ipynb</a></code></td>
      <td>Understanding the Need of Autograd and its Applications</td>
      <td>Autograd, Gradients, Weights and Biases, Autograd Profiler</td>
      <td><code>matplotlib</code>, <code>numpy</code>, <code>torch</code></td>
    </tr>
    <tr>
      <td align=center>4.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/04_building_models_with_pytorch.ipynb>04_building_models_with_pytorch.ipynb</a></code></td>
      <td>Implementing different models and examining various layer types</td>
      <td>Model Parameters, Layers - Linear, Convolutional, Recurrent, Pooling, Normalization, Dropout, Activation and Loss Functions</td>
      <td><code>numpy</code>, <code>torch</code></td>
    </tr>
    <tr>
      <td align=center>5.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/06_training_with_pytorch.ipynb>06_training_with_pytorch.ipynb</a></code></td>
      <td>Training a classifier on the <a href=https://www.kaggle.com/datasets/zalando-research/fashionmnist><code>Fashion-MNIST</code></a> dataset</td>
      <td>Model Training, Training and Validation Loss, Training and Validation Accuracy</td>
      <td><code>matplotlib</code>, <code>sklearn</code>, <code>torch</code>, <code>torchvision</code></td>
    </tr>
    <tr>
      <td align=center>6.</td>
      <td align=center><code><a href=https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/notebooks/07_model_understanding_with_captum.ipynb>07_model_understanding_with_captum.ipynb</a></code></td>
      <td>Using Captum for model interpretability by handling feature and layer attributions</td>
      <td>Model Interpretability, Feature Attributions, Layer Attributions, Captum Insights</td>
      <td><code>captum</code>, <code>matplotlib</code>, <code>numpy</code>, <code>PIL</code>, <code>torch</code></td>
    </tr>
  </tbody>
</table>

### Data

The `data/` directory includes any external data associated with this repository. This contains:

- [`imagenet_class_index.json`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/data/imagenet_class_index.json): Class indices for the ImageNet labels.

### Images

The `images/` directory includes images used in this repository, including:

- [`kabo-p6yH8VmGqxo-unsplash.jpg`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/images/kabo-p6yH8VmGqxo-unsplash.jpg): An image of a cat ([source](https://unsplash.com/photos/orange-tabby-cat-on-yellow-surface-p6yH8VmGqxo)).

### Models

The `models/` directory includes various custom models saved for future reference. The following models can be found:

- [`01_lenet5.pt`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/models/01_lenet5.pt): CNN model based on the LeNet-5 architecture.
- [`04_lstm_tagger.pt`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/models/04_lstm_tagger.pt): LSTM-based RNN model that tags words in a sentence.
- [`06_fashion_clf.pt`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/models/06_fashion_clf.pt): CNN model to classify images from the Fashion MNIST dataset.

### Scripts

The `scripts/` directory contains utility scripts used throughout the tutorial, including:

- [`plot_utils.py`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/scripts/plot_utils.py): Utility functions for tensor operations.
- [`tensor_utils.py`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/scripts/tensor_utils.py): Functions for plotting and visualizing data.
- [`test_utils.py`](https://github.com/DracoY-code/pytorch-tutorial-assets/blob/main/scripts/test_utils.py): Functions to support unit tests.

### Tests

The `tests/` directory includes unit tests for the notebooks and the scripts in the repository. The tests are written using the `pytest` framework to validate the correctness and robustness of the implementations.

The tests can be run by using the following `zsh` script:

```zsh
% ./run_tests.zsh
```

## Resources

- [PyTorch Beginner Series](https://www.youtube.com/playlist?list=PL_lsbAsL_o2CTlGHgMxNrKhzP97BaG9ZN) on YouTube.
- [PyTorch Tutorial Notebooks](https://pytorch.org/tutorials/beginner/basics/intro.html).
