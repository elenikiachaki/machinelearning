#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 22:11:49 2023

@author: annamelidi
"""


import numpy as np
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
np.random.seed(42)

                                ####### Loading the data ###########

# Load the csv data using the Pandas library
my_filename = '/path/to/file/dataR2.csv'
my_df = pd.read_csv(my_filename)

# We will convert the dataframe to numpy arrays 
my_raw_data = my_df.values

# Notice that raw_data both contains the information we want to store in an array
# X1 and the information that we wish to store
# in y1 (the class labels).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the 9 columns from inspecting
# the file.
my_cols = range(0, 9)
X = my_raw_data[:, my_cols]

# We can extract the attribute names that came from the header of the csv
my_attributeNames = np.asarray(my_df.columns[my_cols])

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by
# extracting the strings for each sample from the raw data loaded from the csv:
my_classLabels = my_raw_data[:, -1]  # -1 takes the last column
# Then determine which classes are in the data by finding the set of
# unique class labels
my_classNames = np.unique(my_classLabels)

# Create dictionary to map classes to categories
class_categories = {
    1: 'healthy',
    2: 'cancer'
}

# Map each class to a category
my_classDict = {
    class_label: class_categories[class_label] for class_label in my_classNames}

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is
# assigned. This is the class index vector y:
y = np.array([my_classDict[cl] for cl in my_classLabels])

# We can determine the number of data objects and number of attributes using
# the shape of X1
N, M = X.shape

# Finally, the last variable that we need to have the dataset in the
# "standard representation" for the course, is the number of classes, C1:
C = len(my_classNames)



                            #########   Normalization   ##########
                    

# Calculate the mean and standard deviation of each column
mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)

# Subtract the mean from each element and divide by the standard deviation
X = (X - mean) / std_dev

print("Data were loaded and normalised.\n")
print('============')





#################################   DECISION    TREE    ######################################



    #### 5 fold cross validation to select best alpha and best feautures to train the tree ###

# Define number of folds for cross-validation
k = 5

# Define decision tree classifier with cost complexity pruning
dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=0.0)

# Define k-fold cross-validation generator
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define empty lists to store results
alpha_values = []
train_scores = []
test_scores = []
conf_matrices = []
selected_features_list = []

# Loop over range of alpha values to test
for alpha in np.arange(0.0, 1.0, 0.01):

    # Define empty lists to store results for current alpha
    train_acc = []
    test_acc = []
    conf_mat = np.zeros((2,2))

    # Loop over folds of cross-validation
    for train_idx, test_idx in kf.split(X):

        # Split data into training and test sets for current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit decision tree classifier to training data with current alpha
        dtc = tree.DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=alpha)
        dtc.fit(X_train, y_train)

        # Calculate feature importances using Gini importance
        importances = dtc.feature_importances_

        # Select features with importance greater than 0.1 
        selected_features = np.where(importances > 0.1)[0]
        
        # Check if any feature was selected
        if selected_features.size == 0:
            continue
            
        # Select the subset of features from X_train and X_test
        X_train_selected = X_train[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        # Train decision tree classifier on selected features
        dtc_selected = tree.DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=alpha)
        dtc_selected.fit(X_train_selected, y_train)

        # Calculate training and test accuracy scores for current fold
        y_train_pred = dtc_selected.predict(X_train_selected)
        y_test_pred = dtc_selected.predict(X_test_selected)
        train_acc.append(accuracy_score(y_train, y_train_pred))
        test_acc.append(accuracy_score(y_test, y_test_pred))
        
        # Store selected features for current fold
        selected_features_list.append(selected_features)
        
    # Check if train_acc and test_acc lists are empty
    if not train_acc or not test_acc:
        continue

    # Calculate mean training and test accuracy scores for current alpha
    mean_train_acc = np.mean(train_acc)
    mean_test_acc = np.mean(test_acc)
    
    # Update confusion matrix
    conf_mat += confusion_matrix(y_test, y_test_pred)

    # Append results for current alpha to overall lists
    alpha_values.append(alpha)
    train_scores.append(mean_train_acc)
    test_scores.append(mean_test_acc)
    conf_matrices.append(conf_mat)

# Find alpha with highest test accuracy score
best_alpha_idx = np.argmax(test_scores)
best_alpha = alpha_values[best_alpha_idx]
best_test_acc = test_scores[best_alpha_idx]
best_conf_mat = conf_matrices[best_alpha_idx]

# Find best selected features for best alpha
selected_features_best_alpha = selected_features_list[best_alpha_idx]

# Select the subset of best features for the best alpha from X_train and X_test
X_train_selected = X_train[:, selected_features_best_alpha]
X_test_selected = X_test[:, selected_features_best_alpha]

# Train decision tree classifier on best selected features with the best alpha
dtc_selected = tree.DecisionTreeClassifier(criterion='gini', random_state=42, ccp_alpha=best_alpha)
dtc_selected.fit(X_train_selected, y_train)

# Predict target values for test set
y_test_pred = dtc_selected.predict(X_test_selected)

# Create confusion matrix from test set predictions and actual values
cm_tree = confusion_matrix(y_test, y_test_pred)

# Print results for the best tree
print("Model 1\n","A Decision Tree was fitted to predict the class.", k, "-cross validation was used to train and test the tree.")
print('The best cost complexity parameter (alpha) of the tree is :', best_alpha)
print('and gives the best test accuracy:', best_test_acc, "\n")



# Plot confusion matrix as heatmap with class names
fig, ax = plt.subplots()
sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy","Breast Cancer"], yticklabels=["Healthy","Breast Cancer"], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion matrix of the decision tree')
plt.show()


# Plot alpha values versus training and test accuracy scores
plt.plot(alpha_values, train_scores, label='Training Accuracy')
plt.plot(alpha_values, test_scores, label='Test Accuracy')
plt.xlabel('Alpha')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Alpha values versus training and test accuracy scores")
plt.show()

# Plot the tree
plt.figure(figsize=(50,50))
plot_tree(dtc_selected, feature_names=my_attributeNames, class_names=["Healthy","Breast Cancer"], filled=True, proportion= True)
plt.show()

print("Confussion matrix, alphas vs accuracy and the tree plots were generated.\n")
print('============')






#################################   LOGISTIC REGRESSION    ######################################




# Define number of folds for cross-validation
k = 5

# Define logistic regression classifier
logreg = LogisticRegression(penalty='l2', class_weight='balanced')

# Define k-fold cross-validation generator
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Define empty lists to store results
lambda_values = []
train_scores = []
test_scores = []
conf_matrices = []

# Loop over range of lambda values to test
for lambda_value in np.arange(0.0000000001, 20, 1):
  
    # Define empty lists to store results for current lambda
    train_acc = []
    test_acc = []
    conf_mat = np.zeros((2,2))

    # Loop over folds of cross-validation
    for train_idx, test_idx in kf.split(X):

        # Split data into training and test sets for current fold
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit logistic regression classifier to training data with current lambda
        logreg = LogisticRegression(penalty='l2',  class_weight='balanced', C=1/lambda_value)
        logreg.fit(X_train, y_train)

        # Calculate training and test accuracy scores for current fold
        y_train_pred = logreg.predict(X_train)
        y_test_pred = logreg.predict(X_test)
        train_acc.append(accuracy_score(y_train, y_train_pred))
        test_acc.append(accuracy_score(y_test, y_test_pred))

        # Update confusion matrix
        conf_mat += confusion_matrix(y_test, y_test_pred)

    # Calculate mean training and test accuracy scores for current lambda
    mean_train_acc = np.mean(train_acc)
    mean_test_acc = np.mean(test_acc)

    # Update overall lists
    lambda_values.append(lambda_value)
    train_scores.append(mean_train_acc)
    test_scores.append(mean_test_acc)
    conf_matrices.append(conf_mat)

# Find lambda with highest test accuracy score
best_lambda_idx = np.argmax(test_scores)
best_lambda = lambda_values[best_lambda_idx]
best_test_acc = test_scores[best_lambda_idx]
best_conf_mat = conf_matrices[best_lambda_idx]

# Train logistic regression classifier on full dataset with best lambda
logreg = LogisticRegression(penalty='l2', class_weight='balanced', C=1/best_lambda)
logreg.fit(X_train, y_train)

# Predict target values for test set
y_test_pred = logreg.predict(X_test)

# Create confusion matrix from test set predictions and actual values
cm_log = confusion_matrix(y_test, y_test_pred)

# Plot confusion matrix as heatmap with class names
fig, ax = plt.subplots()
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy","Breast Cancer"], yticklabels=["Healthy","Breast Cancer"], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion matrix of the logistic regression')
plt.show()

# Plot lambda values versus training and test accuracy scores
plt.plot(lambda_values, train_scores, label='Training Accuracy')
plt.plot(lambda_values, test_scores, label='Test Accuracy')
plt.xlabel('Lambda')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Lambda values versus training and test accuracy scores")
plt.show()

# Get coefficients of logistic regression model
coef = logreg.coef_[0]

# Plot coefficients
plt.barh(my_attributeNames, coef)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature Name')
plt.title('Logistic Regression Model Coefficients')
plt.show()

from sklearn.decomposition import PCA

# Fit PCA to the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

# Fit logistic regression model on PCA transformed data
logreg.fit(X_pca[:, :2], y_train)

# Define the range of PC1 and PC2 values
pc1_vals = np.arange(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1, 0.01)
pc2_vals = np.arange(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1, 0.01)

# Create a meshgrid of PC1 and PC2 values
pc1_mesh, pc2_mesh = np.meshgrid(pc1_vals, pc2_vals)

# Predict the class labels for all possible combinations of PC1 and PC2 values
X_mesh = np.column_stack((pc1_mesh.ravel(), pc2_mesh.ravel()))
y_mesh_str = logreg.predict(X_mesh)
y_mesh_int = np.array([list(my_classDict.keys())[list(my_classDict.values()).index(label)] for label in y_mesh_str])
z = y_mesh_int.reshape(pc1_mesh.shape)


# Define a dictionary to map each class string to a number
class_colors = {'healthy': 0, 'cancer': 1}

# Convert the class labels to a sequence of numbers
colors = np.array([class_colors[cl] for cl in y_train])

# Create scatter plot of PC1 and PC2 values colored by class label
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='coolwarm')

# Create contour plot of decision boundary
plt.contourf(pc1_vals, pc2_vals, z, alpha=0.4)

# Add legend
handles, _ = scatter.legend_elements()
labels = [my_classNames[i] for i in range(len(my_classNames))]
plt.legend(handles, my_classDict.values())

# Set axis labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Logistic Regression Decision Boundary')



## Print results for the best logistic regression model
print("Model 2\n","A logistic regression model was fitted to predict the class using", k, "-cross validation.")
print('The best complexity-controlling parameter (lambda) of the model is :', best_lambda)
print('and gives the best test accuracy:', best_test_acc, "\n")
print("Confussion matrix, lambdas vs accuracy, feature ciefficients and decision boundary plots  were generated.\n")

print('============')




#################################   BASELINE   ######################################


# Create the baseline model
baseline_model = DummyClassifier(strategy='most_frequent')

# Evaluate the baseline model using k cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=42)
baseline_scores = cross_val_score(baseline_model, X_train, y_train, cv=kf)

# Compute the confusion matrix for the baseline model
baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)
cm_baseline= confusion_matrix(y_test, y_pred)

# Compute the E

# Plot the training and test accuracy across different folds of cross-validation
plt.figure(figsize=(10, 6))
plt.boxplot([baseline_scores], labels=['Baseline'])
plt.title('Cross-Validation Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.show()

# Plot confusion matrix as heatmap with class names
fig, ax = plt.subplots()
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy","Breast Cancer"], yticklabels=["Healthy","Breast Cancer"], ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion matrix of the baseline model')
plt.show()

print("Model 3\n","A Baseline model was fitted to predict the class using", k, "-cross validation.")

print('============')






##################### EVALUATION OF THE MODELS WITH 2 LEVEL CV   ##############









# Define the parameters for grid search
tree_params = {'ccp_alpha': np.arange(0.0, 1.0, 0.01), 'criterion':['gini']}
logistic_params = {'C': 1/np.arange(0.0000000001, 20, 1), 'penalty': ['l2'], 'class_weight': ['balanced']}


# Define the models
models = {
    'Tree': (DecisionTreeClassifier(), tree_params),
    'Logistic Regression': (LogisticRegression(), logistic_params),
    'Baseline': (DummyClassifier(strategy='most_frequent'), {})
}

# Define the outer cross-validation splits
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the table headers
columns = ['Outer fold', 'Decision Tree Classifier\n alpha','Decision Tree Classifier\n Etest','Logistic Regression\n λ','Logistic Regression\n Etest', 'Baseline\nEtest']

# Create an empty list to store the results
results = []

# Iterate over the outer cross-validation splits
for i, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    # Split the data into training and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create a dictionary to store the results for this fold
    fold_results = {'Outer fold': i+1}

    # Iterate over the models and perform inner cross-validation to select the best hyperparameters
    for name, (model, params) in models.items():
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        search = GridSearchCV(model, params, scoring='accuracy', cv=inner_cv)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        # Store the selected parameter and error rate in the fold_results dictionary
        if name == 'Tree':
            fold_results['Decision Tree Classifier\n alpha'] = search.best_params_['ccp_alpha']
            fold_results['Decision Tree Classifier\n Etest'] = sum(y_pred != y_test) / len(y_test)
        elif name == 'Logistic Regression':
            fold_results['Logistic Regression\n λ'] = 1/search.best_params_['C']
            fold_results['Logistic Regression\n Etest'] = sum(y_pred != y_test) / len(y_test)
        elif name == 'Baseline':
            fold_results['Baseline\nEtest'] = sum(y_pred != y_test) / len(y_test)

    # Append the fold results to the results list
    results.append(fold_results)

# Create a pandas DataFrame from the results list
df = pd.DataFrame(results)

# Set the order of the columns in the DataFrame
df = df[columns]

df.to_excel('Evaluation.xlsx', index=False)  
print("Evaluation of all the models is saved as Evaluation.xlsx.\n")
print('============')





    ###################                    McNemar's test     ########################################

print("Statistical Evaluation.\n")

from scipy.stats import chi2
from scipy import stats

# Create a new DataFrame to store the results
results = pd.DataFrame(columns=['Comparison', 'Null Hypothesis', 'Estimated Difference', 'CI Lower', 'CI Upper', 'p-value', 'Conclusion'])


                    ############    1.  TREE    vs      BASELINE  #################


#calculates the total number of discordant pairs between the two models
n = cm_tree[0, 1] + cm_baseline[1, 0]
#McNemar's test statistic
statistic = float((n - abs(cm_tree[0, 1] - cm_tree[1, 0])) ** 2) / float(n)
p_value = 1 - stats.chi2.cdf(statistic, 1)

# Calculate the difference in estimated generalization error
diff_error = (cm_tree[0,1] + cm_tree[1,0] - cm_baseline[0,1] - cm_baseline[1,0]) / X_test.shape[0]

# Calculate the confidence intervals
se_diff = np.sqrt((cm_tree[0,1]+cm_tree[1,0]+cm_baseline[0,1]+cm_baseline[1,0]) / X_test.shape[0])
alpha = 0.05
ci_lower = diff_error - stats.norm.ppf(1-alpha/2) * se_diff
ci_upper = diff_error + stats.norm.ppf(1-alpha/2) * se_diff



if p_value > alpha:
    conclusion= "Failed to reject null hypothesis: no significant difference in generalization error between the decission tree and the baseline."
    print (conclusion)
else:
    conclusion="Rejected null hypothesis: significant difference in generalization error between tthe decission tree and the baseline."
    print (conclusion)
    
    
print('McNemar Test')
print('---------')
print(f'statistic = {statistic:.4f}')
print(f'p-value = {p_value:.4f}')
print(f'diff_error = {diff_error:.4f}')
print(f'95% Confidence Interval = [{ci_lower:.4f}, {ci_upper:.4f}]')
print('============\n')





                    ############    2.  LOG    vs      BASELINE  #################


#calculates the total number of discordant pairs between the two models
n = cm_log[0, 1] + cm_baseline[1, 0]
#McNemar's test statistic
statistic = float((n - abs(cm_log[0, 1] - cm_log[1, 0])) ** 2) / float(n)
p_value = 1 - stats.chi2.cdf(statistic, 1)

# Calculate the difference in estimated generalization error
diff_error = (cm_log[0,1] + cm_log[1,0] - cm_baseline[0,1] - cm_baseline[1,0]) / X_test.shape[0]

# Calculate the confidence intervals
se_diff = np.sqrt((cm_log[0,1]+cm_log[1,0]+cm_baseline[0,1]+cm_baseline[1,0]) / X_test.shape[0])
alpha = 0.05
ci_lower = diff_error - stats.norm.ppf(1-alpha/2) * se_diff
ci_upper = diff_error + stats.norm.ppf(1-alpha/2) * se_diff



if p_value > alpha:
    conclusion= "Failed to reject null hypothesis: no significant difference in generalization error between the logistic regression and the baseline."
    print(conclusion)
else:
   conclusion= "Rejected null hypothesis: significant difference in generalization error between the logaritmic regression and the baseline."
   print(conclusion)
    
print('McNemar Test')
print('---------')
print(f'statistic = {statistic:.4f}')
print(f'p-value = {p_value:.4f}')
print(f'diff_error = {diff_error:.4f}')
print(f'95% Confidence Interval = [{ci_lower:.4f}, {ci_upper:.4f}]')
print('============\n')





                    ############    3.  LOG    vs      TREE  #################

#calculates the total number of discordant pairs between the two models
n = cm_log[0, 1] + cm_tree[1, 0]
#McNemar's test statistic
statistic = float((n - abs(cm_log[0, 1] - cm_log[1, 0])) ** 2) / float(n)
p_value = 1 - stats.chi2.cdf(statistic, 1)

# Calculate the difference in estimated generalization error
diff_error = (cm_log[0,1] + cm_log[1,0] - cm_tree[0,1] - cm_tree[1,0]) / X_test.shape[0]

# Calculate the confidence intervals
se_diff = np.sqrt((cm_log[0,1]+cm_log[1,0]+cm_tree[0,1]+cm_tree[1,0]) / X_test.shape[0])
alpha = 0.05
ci_lower = diff_error - stats.norm.ppf(1-alpha/2) * se_diff
ci_upper = diff_error + stats.norm.ppf(1-alpha/2) * se_diff



if p_value > alpha:
    print("Failed to reject null hypothesis: no significant difference in generalization error between the logaritmic regression  and the decission tree.")
else:
    print("Rejected null hypothesis: significant difference in generalization error between the  logaritmic regression and the decission tree.") 
    
    
print('McNemar Test')
print('---------')
print(f'statistic = {statistic:.4f}')
print(f'p-value = {p_value:.4f}')
print(f'diff_error = {diff_error:.4f}')
print(f'95% Confidence Interval = [{ci_lower:.4f}, {ci_upper:.4f}]')
print('============\n')














