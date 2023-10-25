# Classification problem using Machine Learning

The purpose of this mini project was to build a model predicting if a breast cancer is benign or malignant based on provided data from examinations. The idea behind it was more a technical implementation of the ML algorithm and verification of its predictive power, rather than focusing on Exploratory Data Analysis (hence choice of dataset). The code was saved as both .py and .ipynb files.

## Input

The dataset comes from UCI Machine Learning Repository and provides information on characteristics of clinical cases. More details can be found [here](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original).

## EDA

All independent variables are integers ranging from 1 to 10, while dependent variable (Class) has two labels: 2 for bening, 4 for malignant. 
There are no missing observations in the dataset.
Independent variables were analyzed by plotting histograms and stacked barplots. Higher values usually resulted in more cases of Class 4. 
Based on correlation matrix, Uniformity of Cell Size was excluded from the model. 
Dimensionality reduction could also be performed, but the dataset is quite small, so performance of the algorithms was sufficient.

## Model building

Dataset was split into test and training set. Feature scaling was also applied, although in this case it was not mandatory (for most of the algorithms).
Following models were tested:
- Logistic regression
- Support Vector Machine
- Random Forest
- Xgboost
- Artificial Neural Networks

In each case, best parameters (increasing accuracy on the training set) were found using grid search. For random forest and Xgboost, feature importance was also examined, showing that Uniformity of Cell Shape was the most important variable in tree splits.

Accuracy score on the test set was compared and based on that, Xgboost was chosen as the final model, achieving 97.8%. For that algorithm, confusion matrix and ROC curve were presented.

## Results

The accuracy of Xgboost model reached 97.8% with area under ROC curve equal to 0.996.
There are only 3 incorrect classifications, False Positives, so the model thought 3 patients had a malignant cancer, but in reality they didn't. 

In addition to accuracy, other metrics are often used to evaluate the model, especially for more imbalanced data. 
For medical predicting (e.g. cancer detection), it is usually better to look at sensitivity, specificity and F measure.
In our case let's check sensitivity, as in general it is better to detect all true cases and live with a few false positives:

Sensitivity = TP/P = 100%.



