import numpy as np
import pandas as pd

# Load the data
Traindata = pd.read_csv('dataset/Mental Health Data/train.csv')
Testdata = pd.read_csv('dataset/Mental Health Data/test.csv')

Traindata['Academic Pressure'] = Traindata['Academic Pressure'].fillna(0)
Testdata['Academic Pressure'] = Testdata['Academic Pressure'].fillna(0)

Traindata['Work Pressure'] = Traindata['Work Pressure'].fillna(0)
Testdata['Work Pressure'] = Testdata['Work Pressure'].fillna(0)

Traindata['Study Satisfaction'] = Traindata['Study Satisfaction'].fillna(0)
Testdata['Study Satisfaction'] = Testdata['Study Satisfaction'].fillna(0)

Traindata['Job Satisfaction'] = Traindata['Job Satisfaction'].fillna(0)
Testdata['Job Satisfaction'] = Testdata['Job Satisfaction'].fillna(0)

Traindata['CGPA'] = Traindata['CGPA'].fillna(Traindata['CGPA'].mean())
Testdata['CGPA'] = Testdata['CGPA'].fillna(Testdata['CGPA'].mean())

Traindata['Dietary Habits'] = Traindata['Dietary Habits'].fillna('Moderate')
Testdata['Dietary Habits'] = Testdata['Dietary Habits'].fillna('Moderate')
Traindata['Financial Stress'] = Traindata['Financial Stress'].fillna(Traindata['Financial Stress'].mean())
Testdata['Financial Stress'] = Testdata['Financial Stress'].fillna(Testdata['Financial Stress'].mean())
Traindata = Traindata.drop(columns=['Profession'])
Testdata = Testdata.drop(columns=['Profession'])
Traindata = Traindata.drop(columns=['Degree'])
Testdata = Testdata.drop(columns=['Degree'])
Traindata = Traindata.drop(columns=['Name'])
Testdata = Testdata.drop(columns=['Name'])
Traindata = Traindata.drop(columns=['id'])
Testdata = Testdata.drop(columns=['id'])
# Convert all string columns to category
# print(Traindata)
# print(Testdata)
for col in Traindata.select_dtypes(include=['object']).columns:
    Traindata[col] = Traindata[col].astype('category')
    Traindata[col] = Traindata[col].cat.codes

for col in Testdata.select_dtypes(include=['object']).columns:
    Testdata[col] = Testdata[col].astype('category')
    Testdata[col] = Testdata[col].cat.codes
    # Traindata[col] = Traindata[col].codes
Traindata.to_csv('dataset/Mental Health Data/processed_train.csv', index=False)
Testdata.to_csv('dataset/Mental Health Data/processed_test.csv', index=False)