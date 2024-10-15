Telco Customer Churn Prediction

This project uses a Multi-Layer Perceptron (MLP) to predict customer churn based on the Telco Customer Churn dataset.

Features

Input data includes features like tenure, TotalCharges, and various categorical features.
The target variable is Churn, indicating if the customer has left the service.
Model

The model is built using TensorFlow, with multiple dense layers and dropout for regularization.
Output is a binary classification using a sigmoid activation function.
Setup

Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction
Install dependencies:
bash
Copy code
pip install tensorflow pandas scikit-learn matplotlib
Run the notebook to preprocess data, train the model, and evaluate results.
License

This project is licensed under the MIT License.
