# making pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


try:
    df = pd.read_csv('loan_prediction.csv')
except FileNotFoundError:
    print('File not found error')
    exit()

df.drop('Loan_ID',axis=1,inplace=True)
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
print(f"Identified {len(numerical_features)} numerical features and {len(categorical_features)} categorical features.")

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #trained and tuned already
    ('classifier', RandomForestClassifier(max_depth= None,
                                          max_features= 'sqrt', 
                                          min_samples_leaf= 4, 
                                          min_samples_split= 2, 
                                          n_estimators= 35))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model_pipeline.fit(X_train, y_train)

import joblib
output_filename = 'loan_prediction_pipeline.joblib'
joblib.dump(model_pipeline, output_filename)
print(f"\nModel pipeline saved to '{output_filename}'")

