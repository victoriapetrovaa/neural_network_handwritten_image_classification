# Neural Network for Handwritten 4 vs. 9 Classification

## Overview

This project implements a neural network in Python for **binary classification of handwritten digits**, with the goal of distinguishing handwritten **4s from handwritten 9s**.

The project demonstrates the complete workflow of a supervised machine-learning classification problem, including data preparation, neural-network initialization, forward propagation, backpropagation, gradient-based learning, model evaluation, and analysis of classification errors.

The neural network was implemented using fundamental numerical operations rather than a high-level machine-learning framework, providing an opportunity to examine the underlying mechanics of neural-network training.

---

## Neural Network Architecture

The model consists of:

* An input layer representing the pixels of each handwritten digit image
* A single hidden layer containing **200 neurons**
* A single output neuron for binary classification
* Randomly initialized weights and biases
* Forward propagation from the input layer through the hidden layer to the output
* Backpropagation using gradients calculated through the chain rule
* Cross-entropy loss for binary classification
* Gradient-based weight updates
* A learning rate of **0.05**

The network uses NumPy matrix operations to process multiple images simultaneously during training.

### Model Structure

```text
Input Layer
    │
    │  W, bH
    ▼
Hidden Layer
200 neurons
    │
    │  U, bO
    ▼
Output Layer
1 neuron
    │
    ▼
Binary Classification
4  ←→  9
```

---

## Dataset

The project uses a dataset of handwritten **4s and 9s** derived from the MNIST handwritten-digit dataset.

The data is stored in:

```text
DATA_mnist_49_3000.mat
```

Each image is represented as a linearized vector of pixel values. The program determines the image dimensions from the number of input features and converts the vectors back into image representations when visualizing examples.

The dataset is divided into training and test sets. The code also checks that the two classes are reasonably balanced in both sets.

---

## Training

The neural network is trained using **cross-entropy loss**, which is appropriate for binary classification.

During training, the cost decreased substantially:

| Training Stage            |   Cost |
| ------------------------- | -----: |
| Initial                   | 0.6917 |
| 250 iterations            | 0.1153 |
| 500 iterations            | 0.0783 |
| 750 iterations            | 0.0653 |
| 1,000 iterations          | 0.0578 |
| Additional 500 iterations | 0.0484 |

The decrease in training cost indicates that the model progressively improved its fit to the training data.

The final reported training cost was:

```text
0.04836
```

---

## Test Results

After training, the network was evaluated on a test set containing **1,000 images**.

The final test cost was:

```text
0.11308
```

The classification results were:

|              | Predicted 4 | Predicted 9 |
| ------------ | ----------: | ----------: |
| **Actual 4** |         471 |          14 |
| **Actual 9** |          29 |         486 |

The model correctly classified:

**957 of 1,000 test images**

This corresponds to:

* **95.7% classification accuracy**
* **4.3% overall error rate**

### Classification Summary

```text
Correctly classified 4s: 471
4s classified as 9s:     14

Correctly classified 9s: 486
9s classified as 4s:     29
```

---

## Misclassification Analysis

In addition to evaluating overall accuracy, the project examines the two types of classification errors separately.

### 4 Classified as 9

There were **14 cases** in which an image whose true class was 4 was incorrectly classified as 9.

Examples of these errors are included in the repository.

### 9 Classified as 4

There were **29 cases** in which an image whose true class was 9 was incorrectly classified as 4.

Examples of these errors are also included in the repository.

Examining individual misclassified images provides a qualitative perspective on the limitations of the model and the visual similarities that can make the two handwritten-digit classes difficult to distinguish.

---

## Examples of Misclassified Images

The repository contains all examples of images that were incorrectly classified by the neural network. A representative sample is included below.

### True 4 → Predicted 9

![True 4 classified as 9](Misclassified%20Images/t0_n1_image1.png)

![True 4 classified as 9](Misclassified%20Images/t0_n1_image2.png)

### True 9 → Predicted 4

![True 9 classified as 4](Misclassified%20Images/t1_n0_image1.png)

![True 9 classified as 4](Misclassified%20Images/t1_n0_image2.png)

![True 9 classified as 4](Misclassified%20Images/t1_n0_image3.png)

---

## Technologies

* **Python**
* **NumPy** — numerical computation and matrix operations
* **SciPy** — loading MATLAB `.mat` data
* **Matplotlib** — visualization of images and training results

---

## Project Structure

```text
Final_Submission/
│
├── Petrova.11_NN_Project.py
├── DATA_mnist_49_3000.mat
├── Misclassified Images/
│   ├── t0_n1_image1.png
│   ├── t0_n1_image2.png
│   ├── t1_n0_image1.png
│   ├── t1_n0_image2.png
│   └── t1_n0_image3.png
│
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/victoriapetrovaa/neural_network_handwritten_image_classification.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the required Python libraries

```bash
pip install numpy scipy matplotlib
```

### 4. Run the project

```bash
python Petrova.11_NN_Project.py
```

The program loads the dataset, trains the neural network, evaluates the trained model on the test set, calculates the classification error rate, and generates visualizations of the results and selected classification errors.

---

## Demo

A video demonstration of the project is available here:

**[Watch the Project Demonstration](INSERT-YOUR-VIDEO-LINK-HERE)**

> **Note:** The demonstration video is hosted separately because the video file is approximately 242 MB and is therefore not stored directly in this GitHub repository.

---

## Key Project Highlights

This project demonstrates practical experience with:

* Binary image classification
* Neural-network architecture
* Forward propagation
* Backpropagation
* Gradient-based optimization
* Cross-entropy loss
* Matrix-based numerical computation
* Training and test-set evaluation
* Quantitative model evaluation
* Classification-error analysis
* Visualization of model results
* Python scientific-computing libraries

---

## Results at a Glance

| Metric                  |              Result |
| ----------------------- | ------------------: |
| Classification task     | Handwritten 4 vs. 9 |
| Hidden-layer neurons    |                 200 |
| Test samples            |               1,000 |
| Correct classifications |                 957 |
| Classification accuracy |           **95.7%** |
| Overall error rate      |            **4.3%** |
| Final training cost     |              0.0484 |
| Test cost               |              0.1131 |

---

## Future Improvements

Potential extensions to this project could include:

* Experimenting with different hidden-layer sizes
* Comparing alternative activation functions
* Evaluating different learning rates
* Performing systematic hyperparameter tuning
* Comparing the neural network with other classification approaches
* Expanding the classification task to additional handwritten digits
* Investigating the characteristics of images that are consistently difficult to classify

---

## Author

**Victoria Petrova**

[LinkedIn](https://www.linkedin.com/in/victoria-petrovaa/) · [GitHub](https://github.com/victoriapetrovaa)
