import pandas as pd
from sklearn import datasets
data = datasets.load_breast_cancer()

dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])
dataset

from sklearn.model_selection import train_test_split
X = dataset .copy()
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X_train, y_train)

clf.get_params()

X_test

predictions = clf.predict(X_test)
predictions


clf.predict_proba(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions, labels=[0,1])

from sklearn.metrics import precision_score
precision_score(y_test, predictions)

from sklearn.metrics import recall_score
recall_score(y_test, predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['malignant','benign']))

feature_names = X.columns
feature_names

clf.feature_importances_

feature_importance = pd.DataFrame(clf.feature_importances_,index = feature_names).sort_values(0,....incomplete

features = list(feature_importance[feature_importance[0]>0].index)
features

feature_importance.head(10).plot(kind='bar')

from sklearn import tree
from matplotlib import pyplot as plt

fig = plt.figure(figsize=(25,20))
_=tree.plot_tree(clf,

feature_names=feature_names,class_names={0: 'Malignant', 1: 'Benign'},filled=True,fontsize =12}

