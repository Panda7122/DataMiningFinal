import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
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
# val_X, X_test, val_Y, y_test = train_test_split(X_test, y_test, test_size = 0.6, random_state = _random_state)
# Initialize the AdaBoost classifier
ada_clf = AdaBoostClassifier(random_state=_random_state, n_estimators=_n_estimators)
ada_clf.fit(X_train, y_train)


# Make predictions
y_pred = ada_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
import matplotlib.pyplot as plt

# Compute the ROC curve and AUC
y_prob = ada_clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('adaboostROC.png')

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

yAns = ada_clf.predict(testdata)
testdata = pd.read_csv('dataset/Mental Health Data/test.csv')

with open("adaboostOutput.csv", 'w') as f:
    f.write('id,Depression\n')
    for i, idx in enumerate(testdata['id']):
        f.write(f'{idx}, {yAns[i]}\n')