
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR # Import the SVR model
import joblib
import warnings

warnings.filterwarnings("ignore")

try:
    df = pd.read_csv('medical_cost_insurance.csv')
except FileNotFoundError:
    print("Error: 'medical_cost_insurance.csv' not found.")
    exit()


X = df.drop('charges', axis=1)
y = df['charges']

categorical_features = ['sex', 'smoker', 'region']
numerical_features = ['age', 'bmi', 'children']


numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    #trained already
    ('svr', SVR(C=4, epsilon=0.12, gamma='scale', kernel='rbf'))
])

model_pipeline.fit(X, y)

joblib.dump(model_pipeline, 'med_insurance_svr_pipeline.joblib')
print("Pipeline saved to 'med_insurance_svr_pipeline.joblib'.")