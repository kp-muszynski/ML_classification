
"""## 0) Import libraries and the dataset"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Dataset: https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original
df = pd.read_csv('Data.csv')
df = df.drop('Sample code number', axis=1)

"""## 1) Base statistics of the dataset"""

df.info()

df.describe().T

df.head()

df.groupby(["Class"]).agg("mean") # Class 2 means cancer is benign, while 4 is malignant

"""## 2) Initial analysis"""

# Let's draw some histograms

df.hist(bins=30, figsize=(15, 10))

# Since all columns other than Class have integer values between 1 and 10, I will draw stacked barplots

# cross tab as base for barplot
def crosstab_generator(df, column):
  return pd.crosstab(index=df[column],
                        columns=df['Class'],
                        normalize="index")

# function drawing bar plot for each column
def stacked_plot(df, item, i, j):
  dataset = crosstab_generator(df, item)

  value_counts = list(df[item].value_counts().sort_index())

  ax = dataset.plot(kind='bar',
                    stacked=True,
                    colormap='tab10',
                    ax=axes[i][j])

  # adding labels to bars
  for c in ax.containers:
    labels = [v.get_height() for v in c]
    labels = [int(round(labels[index]*l_item,0))  if labels[index]>0 else '' for index, l_item in enumerate(value_counts)]
    ax.bar_label(c, labels=labels, label_type='center')

# Let's populate the plots

fig, axes = plt.subplots(nrows=3,ncols=3)
fig.set_size_inches(18, 12)

col_list = df.columns.values.tolist()
col_list.remove('Class')

for index, item in enumerate(col_list):
  j = index % 3
  i = index // 3

  stacked_plot(df, item, i, j)

# Now let's look at correlation matrix
corrMatt = df.corr()

mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = 0

fig,ax= plt.subplots()
fig.set_size_inches(10,8)
sn.heatmap(corrMatt,mask=mask, vmax=.8, square=True, annot=True)

"""## 3) Data preprocessing"""

# I'll remove column Uniformity of cell size, as it is highly (>0.8) correlated with Uniformity of Cell Shape

X = df.iloc[:, [0,2,3,4,5,6,7,8]].values
y = df.iloc[:,-1].values

X_cols = df.iloc[:, [0,2,3,4,5,6,7,8]].columns

# There is no missing data and no encoding is needed, as values are already between 1 and 10
print(df.isna().sum())

# Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling (even though in this case it was not mandatory for most of the algorithms)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""## 4) Building models

#### 4.1) Logistic regression
"""

# logistic regression model
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Let's once check confusion matrix of default model
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# grid search to find best parameters
from sklearn.model_selection import GridSearchCV
parameters =[{"C":[0.1, 0.25, 0.5, 0.75, 1]},
              {"C":[0.1, 0.25, 0.5, 0.75, 1], "penalty":["l1","l2"], "solver":["liblinear"]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

"""#### 4.2) Support Vector Machine"""

# SVM model
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# grid search to find best parameters
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

"""#### 4.3) Random forest"""

# Random forest
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Feature importance of default model
feature_imp = pd.Series(classifier.feature_importances_, index=X_cols).sort_values(ascending=False)

sn.barplot(x=feature_imp, y=feature_imp.index)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Random Forest feature importance')
plt.show()

# grid search to find best parameters
parameters = {'n_estimators': [100],
              'criterion': ['entropy'],
              'max_depth': [3, 5, 7, 9, 11, 13],
              'min_samples_split': [2, 5, 8, 11],
              'max_features': [2, 3, 4, 5, 6],
              'min_samples_leaf': [1, 2, 3]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Feature importance of the best model
classifier_rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,
                                       max_depth = 3, min_samples_split = 11, max_features = 3, min_samples_leaf = 3)
classifier_rf.fit(X_train, y_train)

feature_imp_rf = pd.Series(classifier_rf.feature_importances_, index=X_cols).sort_values(ascending=False)

sn.barplot(x=feature_imp_rf, y=feature_imp_rf.index)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Random Forest feature importance')
plt.show()

"""#### 4.4) Xgboost"""

# Xgboost
from sklearn.ensemble import GradientBoostingClassifier

classifier = GradientBoostingClassifier(n_estimators = 100, random_state = 0)
classifier.fit(X_train, y_train)

# grid search to find best parameters
parameters = {'n_estimators': [100],
              'learning_rate': [0.01, 0.05, 0.1],
              'subsample': [0.8],
              'max_depth': [3, 5, 7, 9, 11, 13],
              'min_samples_split': [2, 5, 8, 11],
              'max_features': [2, 3, 4, 5, 6],
              'min_samples_leaf': [1, 2, 3]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

# Feature importance of the best model
classifier_xg = GradientBoostingClassifier(n_estimators = 100, subsample = 0.8, random_state = 0,
                                       max_depth = 13, min_samples_split = 11, max_features = 2, min_samples_leaf = 1, learning_rate = 0.1)
classifier_xg.fit(X_train, y_train)

feature_imp_xg = pd.Series(classifier_xg.feature_importances_, index=X_cols).sort_values(ascending=False)

sn.barplot(x=feature_imp_xg, y=feature_imp_xg.index)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('XGboost feature importance')
plt.show()

"""#### 4.5) ANN"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y_train_ann = le.fit_transform(y_train)
y_test_ann = le.transform(y_test)

import tensorflow as tf

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train_ann, batch_size = 30, epochs = 100)

"""
## 5) Model selection"""

# make a list of models
models = []
models.append(('Logistic regression', LogisticRegression(random_state = 0, C = 0.1)))
models.append(("SVM", SVC(kernel = 'linear', random_state = 0, C = 0.25)))
models.append(("Random forest", RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0,
                                       max_depth = 3, min_samples_split = 11, max_features = 3, min_samples_leaf = 3)))
models.append(("Xgboost", GradientBoostingClassifier(n_estimators = 100, subsample = 0.8, random_state = 0,
                                       max_depth = 13, min_samples_split = 11, max_features = 2, min_samples_leaf = 1, learning_rate = 0.1)))

# Verify accuracy on test set for all models
for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        msg = "%s: (%f)" % (name, accuracy)
        print(msg)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print("ANN" + accuracy_score(y_test_ann, y_pred))

# Making the Confusion Matrix for chosen model (Xgboost, as it has the highest accuracy)
y_pred = classifier_xg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# FN = cm[1, 0]; Positives = 0+50

tn, fp, fn, tp = cm.ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)

"""## 6) ROC curve

"""

# Let's plot the ROC curve of the chosen model
from sklearn.metrics import roc_curve, auc

y_pred_prob = classifier_xg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label = 4)
roc_auc = auc(fpr,tpr)

fig, ax = plt.subplots()
ax.plot(fpr,tpr, label = " area = {:0.3f}".format(roc_auc))
ax.plot([0,1], [0,1], 'r', linestyle = "--", lw = 2)
ax.set_xlabel("False Positive Rate", fontsize = 10)
ax.set_ylabel("True Positive Rate", fontsize = 10)
ax.set_title("ROC Curve", fontsize = 15)
ax.legend(loc = 'best')
