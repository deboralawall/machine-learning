# Artificial Neural Network for Multi-Class Classification

This project explores a supervised machine learning workflow for classifying synthetic 2D data using artificial neural networks (ANNs). The repository includes two Jupyter notebooks and two corresponding datasets:

- `main4.ipynb` with `Base4.csv`
- `main15.ipynb` with `Base15.csv`

Each dataset contains two numeric feature columns and one target label column, representing a multi-class classification problem. The goal is to train a neural network that learns how to assign each observation to the correct class based on its coordinates.

## Project Objective

The main objective is to build, train, and evaluate a multilayer perceptron (MLP) that can correctly classify observations into multiple categories. The project follows a standard machine learning pipeline:

1. Understand the data and define features versus target.
2. Perform exploratory data analysis (EDA).
3. Scale the numerical attributes.
4. Split the data into training, validation, and test sets.
5. Build a neural network in Keras.
6. Train the model and monitor convergence.
7. Evaluate the model with classification metrics.

## Dataset Structure

Both files are structured as a tabular dataset with three columns:

- Column 0: feature 1
- Column 1: feature 2
- Column 2: class label

The data is synthetic and designed to represent grouped spatial patterns, which makes it suitable for demonstrating how neural networks can separate classes in a two-dimensional space.

## Workflow

### 1. Data loading and preprocessing

The notebooks read the CSV files, clean the values, and convert them into a DataFrame. The features are selected from the first two columns, while the label is taken from the third column.

The target values are adjusted so that classes start at index 0, matching the expected format used by Keras for sparse categorical classification.

### 2. Exploratory Data Analysis

The notebooks visualize the data using:

- Histograms
- Box plots by class
- Correlation matrix
- Optional plotting of class separation

This analysis helps understand the distribution of the variables and check whether the classes are clearly distinguishable.

### 3. Feature scaling

Since neural networks are sensitive to feature magnitude, the project uses `MinMaxScaler` to normalize the input variables into a common range. This improves numerical stability and training efficiency.

### 4. Train/validation/test split

The dataset is divided into three subsets:

- Training set: used to learn model parameters
- Validation set: used to tune learning behavior and detect overfitting
- Test set: used for the final unbiased evaluation

The notebooks use a two-step split to create the three sets while preserving a balanced workflow for model assessment.

### 5. Neural network architecture

The implemented model is a Keras sequential MLP with:

- Input layer with 2 features
- Hidden layer 1 with ReLU activation
- Hidden layer 2 with ReLU activation
- Output layer with softmax activation

The number of output neurons matches the number of classes in the dataset:

- 4 classes in `Base4.csv`
- 15 classes in `Base15.csv`

The optimizer used is Adam, and the loss function is `sparse_categorical_crossentropy`, which is appropriate for integer class labels.

### 6. Training and monitoring

The model is trained for a fixed number of epochs, and `EarlyStopping` is used to prevent unnecessary training when validation loss stops improving. Each notebook plots the training history to show how accuracy and loss evolve over time, which helps explain the learning dynamics of the ANN.

### 7. Model evaluation

After training, the notebooks compute key classification metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These metrics are used to assess how well the model generalizes to unseen data and how effectively it distinguishes among the classes.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Repository Structure

```text
.
├── Base4.csv
├── Base15.csv
├── main4.ipynb
├── main15.ipynb
├── ML-ANN-Classification.pdf
└── README.md
```

## How to Run

1. Install the required dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

2. Open the notebook you want to run:

```bash
jupyter notebook main4.ipynb
```

or

```bash
jupyter notebook main15.ipynb
```

3. Execute the cells in order to load the dataset, train the model, and evaluate the results.

## Notes

This project is intended as a practical example of how to apply neural networks to a multi-class classification problem using tabular data. It is useful for learning about preprocessing, model design, validation, and evaluation in a deep learning context.
