# Restaurant-Recommendation

# Hybrid Restaurant Recommendation System

## 🍽️ Overview
An intelligent restaurant recommendation system that combines content-based filtering with machine learning to provide personalized dining suggestions. The system uses a hybrid approach, leveraging both user preferences and advanced ML algorithms to deliver accurate recommendations.

## 🚀 Features
- Hybrid recommendation engine (Content-Based + Random Forest)
- Automated data preprocessing and feature engineering
- Multiple evaluation metrics
- Customizable user preference system
- Restaurant scoring based on multiple parameters

## 🛠️ Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## 📊 Algorithm Components
1. **Content-Based Filtering**
   - OneHotEncoder for categorical variables
   - Cosine Similarity for preference matching

2. **Machine Learning**
   - Random Forest Classifier
   - StandardScaler for feature normalization
   - Train-test split validation

## 📈 Performance Metrics
- Accuracy
- Precision
- Recall
- F1-Score

## 🔧 Installation
bash
# Create and activate virtual environment
python -m venv restaurant_env

# For Windows
restaurant_env\Scripts\activate

# For macOS/Linux
source restaurant_env/bin/activate

# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn
pip install jupyter notebook

## 🖥️ Usage
# Load the Dataset
import pandas as pd
df = pd.read_csv('path_to_your_dataset.csv')

# Preprocess the Data
 Your preprocessing code  here 

# Train the Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
-- Your training code here

# Make Predictions
predictions = model.predict(new_data)

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

