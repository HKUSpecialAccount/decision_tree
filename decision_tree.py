import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import sys
from graphviz import Graph
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
try:
    data = pd.read_csv('loan_approval_dataset.csv')
except:
    print("file not found")
    sys.exit()
X = data.drop('loan_status', axis=1)
y = data['loan_status']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = DecisionTreeClassifier(max_depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
accuracy1 = accuracy_score(y_train, y_pred)
y_pred = model.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred)
print('Accuracy on training set: ', accuracy1)
print(accuracy1*len(X_test))
print('Accuracy on testing set: ',accuracy2)
plt.figure(figsize=(30,20))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['approved', 'rejected'])
plt.show()
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
