# 🩸Anemia Types Classification EDA&Prediction Keras

This project aims to classify different types of anemia using a dataset containing various blood test parameters. The classification is achieved using a deep learning model built with Keras.

## 📊 Dataset
The dataset used in this project is the "Diagnosed CBC Data" which contains various blood test parameters and their corresponding anemia types. The dataset is available at [Kaggle](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification).

## ⚙️ Technologies Used
- 🐍 Python
- 🐼 Pandas
- 📊 NumPy
- 📊 Matplotlib
- 📈 Seaborn
- 🤖 Scikit-learn
- 🤖 TensorFlow (Keras)

## 🏗️ Model Architecture
The model is a sequential neural network with the following architecture:
- Input layer with 128 neurons and ReLU activation
- Batch Normalization
- Dropout layer with a rate of 0.3
- Hidden layers with 64, 32, 16, and 8 neurons, each followed by Batch Normalization and Dropout
- Output layer with softmax activation for multi-class classification

### ⏳ Early Stopping
Early stopping is implemented to prevent overfitting by monitoring the validation loss.

## 📈 Results
The model is evaluated on a test set, and the accuracy is reported. The training and validation loss and accuracy are visualized over epochs.

## 📉 Visualizations
Several visualizations are included in the project:
- Distribution of diagnoses
- Histograms of numerical features
- Boxplots for outlier analysis
- Correlation matrix heatmap
- Confusion matrix for model predictions
- ROC curves for each class

## 📜 License

This project is licensed under the [MIT](LICENSE) License.

## 🔍 Code And Kaggle Link
Project: [anemia-types-classification-eda-prediction-keras.ipynb](https://github.com/omerfarukyuce/Anemia-Types-Classification-EDA-Prediction-Keras/blob/main/anemia-types-classification-eda-prediction-keras.ipynb)

Kaggle: [🩸Anemia Types Classification EDA&Prediction Keras)](https://www.kaggle.com/code/merfarukyce/anemia-types-classification-eda-prediction-keras/notebook)

## 📊 Datasets
Dataset: [Kaggle](https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification).
