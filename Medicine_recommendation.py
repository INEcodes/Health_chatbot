import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('C:/Users/Abhay/Documents/GitHub/Health_chatbot/synthetic_medical_data.csv')

print(data.head())

data = pd.get_dummies(data, columns=['symptoms', 'diagnosis'])

X = data.drop('recommended_medicine', axis=1)
y = data['recommended_medicine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)