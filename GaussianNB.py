import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
_random_state = 42
_n_estimators = 10000
# Load the dataset
data = pd.read_csv('dataset/Mental Health Data/processed_train.csv')
testdata = pd.read_csv('dataset/Mental Health Data/processed_test.csv')

# Separate the features and the target variable
X = data.drop('Depression', axis=1)
y = data['Depression']
# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=_random_state)

# Initialize the Gaussian Naive Bayes model
model = GaussianNB(var_smoothing=1e-8)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate classification report
report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('GaussianNBROC.png')

y_ans = model.predict(testdata)
testdata = pd.read_csv('dataset/Mental Health Data/test.csv')

with open("GaussianNBOutput.csv", 'w') as f:
    f.write('id,Depression\n')
    for i, idx in enumerate(testdata['id']):
        f.write(f'{idx}, {y_ans[i]}\n')

