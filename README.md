# Graduate Admission Prediction using ANN

An Artificial Neural Network (ANN) built to predict the probability of a student's admission to a graduate program based on academic and extracurricular credentials. This project frames the prediction as a regression problem, outputting a continuous probability score.

## Project Overview

The objective of this project is to evaluate student profiles—including GRE Scores, TOEFL Scores, University Ratings, Statement of Purpose (SOP), Letter of Recommendation (LOR) strength, CGPA, and Research Experience—to accurately predict the likelihood of acceptance. 

## Technology Stack

* Language: Python 3.x
* Deep Learning Framework: TensorFlow / Keras
* Data Manipulation & Analysis: Pandas, NumPy
* Data Visualization: Matplotlib

## Preprocessing & Core Concepts

### 1. Feature Scaling (MinMax Scaling)
Neural networks use Gradient Descent to update weights. When features have vastly different ranges (e.g., GRE scores range up to 340, while CGPA ranges up to 10), the gradient updates can oscillate wildly, slowing down convergence. 

To resolve this, MinMax Scaling is applied to bound all numerical features strictly between 0 and 1 using the formula:

$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

This ensures smooth optimization paths during backpropagation.

### 2. Structural Regularization
For small datasets (such as this 500-row dataset), high-capacity deep learning models are prone to overfitting (memorizing the training data). Rather than applying explicit dropout or weight penalties, this project uses structural regularization by maintaining a highly constrained, shallow architecture (only 120 total trainable parameters) to force the network to learn generalized patterns.

## Model Architecture

The model is built using a Multi-Layer Perceptron (MLP) architecture:

* **Input Layer:** Accepts 7 features after dropping the identifier column (`Serial No.`).
* **Hidden Layer 1:** Dense layer with 7 neurons and ReLU activation function ($f(x) = \max(0, x)$).
* **Hidden Layer 2:** Dense layer with 7 neurons and ReLU activation function.
* **Output Layer:** A single neuron with a linear activation function to predict the continuous target `Chance of Admit`.
* **Loss Function:** Mean Squared Error (MSE), which penalizes larger errors more heavily by squaring the residuals:

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **Optimizer:** Adam (Adaptive Moment Estimation).

## Dataset

The model is trained on the Graduate Admissions Dataset.
* **Features:** GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research (Binary).
* **Target Variable:** Chance of Admit (Continuous value between 0 and 1).

## Implementation Details

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load Data
df = pd.read_csv('Admission_Predict_Ver1.1.csv')
df.drop(columns=['Serial No.'], inplace=True)

X = df.iloc[:, 0:-1]
y = df.iloc[:,-1]

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build Model
model = Sequential()
model.add(Dense(7, activation='relu', input_dim=7))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear'))
```

## Results and Evaluation
Training and Validation loss converged stably over 100 epochs without divergence.

R-squared (Coefficient of Determination) Score: ~0.795, meaning approximately 79.5% of the variance in the admission probability is predictable from the input metrics.

## License
This project is licensed under the MIT License.
# Compile and Train
model.compile(loss='mean_squared_error', optimizer='Adam')
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)
