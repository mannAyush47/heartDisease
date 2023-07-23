# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# Data preprocessing
dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# Split the dataset into features (X) and target variable (y)
y = dataset['target']
X = dataset.drop(['target'], axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-nearest neighbors classifier
knn_scores = []
for k in range(1, 21):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn_classifier, X_train, y_train, cv=10)
    knn_scores.append(score.mean())

# Plot KNN scores for different K values
plt.plot([k for k in range(1, 21)], knn_scores, color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i - 1], (i, knn_scores[i - 1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

# Choose the best K value for KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=12)
knn_classifier.fit(X_train, y_train)

# Evaluate KNN classifier
knn_accuracy = accuracy_score(y_test, knn_classifier.predict(X_test))
print("KNN Accuracy:", knn_accuracy)

# Random Forest classifier
randomforest_classifier = RandomForestClassifier(n_estimators=10)
randomforest_classifier.fit(X_train, y_train)

# Evaluate Random Forest classifier
rf_accuracy = accuracy_score(y_test, randomforest_classifier.predict(X_test))
print("Random Forest Accuracy:", rf_accuracy)

# Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

# Evaluate Decision Tree classifier
dt_accuracy = accuracy_score(y_test, dt_classifier.predict(X_test))
print("Decision Tree Accuracy:", dt_accuracy)

# Make predictions on new data points
new_data = pd.DataFrame({
    'age': [55],
    'sex': ['male'],
    'cp': ['typical'],
    'trestbps': [130],
    'chol': [250],
    'fbs': ['false'],
    'restecg': ['normal'],
    'thalach': [150],
    'exang': ['false'],
    'oldpeak': [2.5],
    'slope': ['flat'],
    'ca': [1],
    'thal': ['fixed']
})

# Preprocess the new data
new_data = pd.get_dummies(new_data, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
new_data[columns_to_scale] = scaler.transform(new_data[columns_to_scale])

# Use the trained classifiers to make predictions
knn_prediction = knn_classifier.predict(new_data)
rf_prediction = randomforest_classifier.predict(new_data)
dt_prediction = dt_classifier.predict(new_data)

# Interpret the predictions
if knn_prediction == 1:
    print("KNN predicts chances of heart disease.")
else:
    print("KNN predicts no chances of heart disease.")

if rf_prediction == 1:
    print("Random Forest predicts chances of heart disease.")
else:
    print("Random Forest predicts no chances of heart disease.")

if dt_prediction == 1:
    print("Decision Tree predicts chances of heart disease.")
else:
    print("Decision Tree predicts no chances of heart disease.")
