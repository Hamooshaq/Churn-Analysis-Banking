Telco Customer Churn Prediction

This repository contains a deep learning model to predict customer churn using a Multi-Layer Perceptron (MLP) implemented in TensorFlow/Keras. The model is designed to classify whether a customer will churn (leave) or not, based on various features from the Telco Customer Churn dataset.

Project Overview

This project aims to:

Preprocess the data by cleaning and transforming categorical features.
Implement a Multi-Layer Perceptron (MLP) model with TensorFlow for binary classification.
Train and evaluate the model using the churn dataset.
Visualize the training progress with accuracy and loss metrics.
Dataset

The dataset used in this project is the Telco Customer Churn dataset. It contains information about customers, their demographic details, account information, and whether or not they have churned.

Key Features:
TotalCharges: Total amount charged to the customer.
tenure: Number of months the customer has stayed with the company.
Various binary and categorical features such as Contract, InternetService, PaymentMethod.
Target Variable:
Churn: Whether the customer has churned (Yes/No).
Requirements

To run this project, you will need the following Python libraries:

bash
Copy code
pip install tensorflow pandas scikit-learn matplotlib
How to Run

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/telco-churn-prediction.git

cd telco-churn-prediction

Preprocessing the data: The notebook handles various preprocessing steps such as:
Dropping less relevant columns like customerID, gender, etc.
Converting the TotalCharges column to numeric.
Handling missing values and encoding categorical variables.
Training the Model: The MLP model is initialized with multiple dense layers with ReLU activation functions and a sigmoid output layer for binary classification. Here's the core structure of the model:
python
Copy code

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

Training and Evaluation:
The model is trained using the training dataset, with validation data for monitoring.
Training accuracy and loss are visualized to evaluate model performance.
Visualization: After training, the model's performance is visualized using matplotlib to plot:
Training vs Validation Accuracy
Training vs Validation Loss
Results

The model achieves reasonable performance on the churn dataset, and its effectiveness is evaluated through metrics like accuracy and loss.

File Structure

case.ipynb: The main Jupyter Notebook containing the model, preprocessing steps, and evaluation.
Telco-Customer-Churn.csv: The dataset used for training the model (ensure it is placed in the correct path).
Contributing

If you wish to contribute to this project, feel free to fork the repository and submit pull requests.

License

This project is licensed under the MIT License.
