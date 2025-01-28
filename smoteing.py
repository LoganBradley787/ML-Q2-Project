import pandas as pd
from imblearn.over_sampling import SMOTE

train = pd.read_csv('train.csv')

X_train = train.drop('class', axis=1)
y_train = train['class']

print("Starting class distribution:")
print(y_train.value_counts())

smote = SMOTE(random_state=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

smoted_train = pd.concat([X_train_smote, y_train_smote], axis=1)

smoted_train.to_csv('train_smote.csv', index=False)

print("Ending class distribution:")
print(y_train_smote.value_counts())