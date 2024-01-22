
import numpy as np
import pandas as pd
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel,xticks, clim, yticks,legend,show, suptitle, title)
import torch
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from toolbox_02450 import *
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)



from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


# Load the dataset into a Pandas DataFrame
df = pd.read_csv('/path/to/file/dataR2.csv')

# Select the predictor variables (X) and the target variable (y)
X = df.iloc[:, [3, 5, 7, 8]]
y = df.iloc[:, 4]

# Standardize the predictor variables
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define a range of lambda values to test
lambdas = np.logspace(-3, 3, num=7)

# Set up k-fold cross-validation with k=10
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform k-fold cross-validation for each value of lambda
train_errors = []
test_errors = []
for lambd in lambdas:
    ridge = Ridge(alpha=lambd)
    errors_train = []
    errors_test = []
    for train_index, test_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        ridge.fit(X_train, y_train)
        errors_train.append(np.mean((ridge.predict(X_train) - y_train) ** 2))
        errors_test.append(np.mean((ridge.predict(X_test) - y_test) ** 2))
    train_errors.append(np.mean(errors_train))
    test_errors.append(np.mean(errors_test))

# Find the lambda with minimum test error
optimal_lambda = lambdas[np.argmin(test_errors)]

# Plot the train and test errors for different values of lambda
plt.semilogx(lambdas, test_errors, label='Validation error')
plt.semilogx(lambdas, train_errors, label='Train error')
plt.xlabel('Lambda')
plt.ylabel('Squared error')
plt.legend()
plt.text(optimal_lambda, 5, f'Optimal lambda: {optimal_lambda:.2e}', ha='center', va='top')
plt.show()
ridge = Ridge(alpha=optimal_lambda)
ridge.fit(X, y)
print(ridge.coef_)
######################################### Load data ###########################################

# Load the csv data using the Pandas library

my_filename = '/path/to/file/dataR2.csv'
my_df = pd.read_csv(my_filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = my_df.values  


# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the 10 columns from inspecting 
# the file.
cols =[3, 5, 7, 8]
X = raw_data[:, cols]
y = raw_data[:, 4]
y = y.reshape(-1, 1)

N, M = X.shape

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(my_df.columns[cols])

print("Data were loaded.\n")


###############################################################################################
                            #########   Normalization   ##########
                    

# Calculate the mean and standard deviation of each column
mean = np.mean(X, axis=0)
std_dev = np.std(X, axis=0)

# Subtract the mean from each element and divide by the standard deviation
X = (X - mean) / std_dev

print("Data were loaded and normalised.\n")
print('============')




#################################   BASELINE   ##############################################


# Create the baseline model
baseline_model = DummyRegressor(strategy='mean', constant=None, quantile=None)


#################################   LINEAR MODEL FOR λ = 1   ##############################################


import numpy as np
from sklearn.linear_model import Ridge



# Initialize Ridge regression model with regularization parameter lambda
linearmodel1 = Ridge(alpha= 1)

# Fit model to data
linearmodel1.fit(X, y)



################################### ANN 3 fold cross validation  ########################################################
###### This script was ran for multiple h and max_iter values, to find the optimal ones#########
# Parameters for neural network classifier
n_hidden_units = 2      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 10000

# K-fold crossvalidation
K = 3                   # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model
model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.Tensor(X[train_index,:])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index,:])
    y_test = torch.Tensor(y[test_index])
    
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold 
    
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Alcohol content: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()
#################################   ΑΝΝ FOR h = 3   ##############################################




def ANN(X, y, h=3, max_iter=10000):
    # Convert data to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().reshape(-1, 1)
    
    # Define the model
    model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1)
            )
    # Initialize model and loss function
    net3 = model()
    loss_fn = torch.nn.MSELoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(net3.parameters())
    
    # Train the model
    for i in range(max_iter):
        y_pred = net3(X)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print("Iteration: {} - Loss: {}".format(i, loss.item()))
    
    # Return the trained model
    return net3




##################### EVALUATION OF THE BASELINE MODEL WITH 2 LEVEL CV   ###########################

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

# Define the outer and inner cross-validation folds
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)


# Initialize the baseline model
baseline_model = DummyRegressor(strategy='mean')

# Initialize a list to store the MSEs for each outer fold
outer_mses = []

# Perform the outer cross-validation loop
for train_outer, test_outer in outer_cv.split(X):
    # Split the data into training and testing sets for the outer fold
    X_train_outer, X_test_outer = X[train_outer], X[test_outer]
    y_train_outer, y_test_outer = y[train_outer], y[test_outer]
    
    # Initialize a list to store the MSEs for each inner fold
    inner_mses = []
    
    # Perform the inner cross-validation loop
    for train_inner, test_inner in inner_cv.split(X_train_outer):
        # Split the data into training and testing sets for the inner fold
        X_train_inner, X_test_inner = X_train_outer[train_inner], X_train_outer[test_inner]
        y_train_inner, y_test_inner = y_train_outer[train_inner], y_train_outer[test_inner]
        
        # Train the baseline model on the inner training set
        baseline_model.fit(X_train_inner, y_train_inner)
        
        # Evaluate the baseline model on the inner testing set
        y_pred_inner = baseline_model.predict(X_test_inner)
        inner_mses.append(mean_squared_error(y_test_inner, y_pred_inner))
    
    # Take the average of the inner MSEs as the performance metric for the outer fold
    outer_mse = np.mean(inner_mses)
    outer_mses.append(outer_mse)

# Print the MSEs for each outer fold
print("MSEs for each outer fold:", outer_mses)

# Calculate and print the mean and standard deviation of the MSEs over all outer folds
mean_mse = np.mean(outer_mses)
std_mse = np.std(outer_mses)
print("Mean MSE over all outer folds:", mean_mse)
print("Standard deviation of MSE over all outer folds:", std_mse)




###############################################################################################

                     # EVALUATION OF LINEAR REGRESSION MODEL WITH 2 LEVEL CV #
                                     
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge


# Define the lambda values to be used in the ridge regression
lambdas = [0.01, 0.1 , 1, 10, 100, 1000, 10000]

# Define the outer and inner folds
outer_kfold = KFold(n_splits=10, shuffle=True, random_state=42)
inner_kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize variables
K_outer = outer_kfold.get_n_splits()
K_inner = inner_kfold.get_n_splits()
lambdas = np.array([0.01, 0.1, 1, 10, 100, 1000, 10000])
errors = np.zeros((K_outer, lambdas.shape[0], K_inner))

# Outer cross-validation loop
for i, (train_outer, test_outer) in enumerate(outer_kfold.split(X)):
    X_train_outer, y_train_outer = X[train_outer], y[train_outer]
    X_test_outer, y_test_outer = X[test_outer], y[test_outer]
    
    # Inner cross-validation loop
    for j, (train_inner, val) in enumerate(inner_kfold.split(X_train_outer, y_train_outer)):
        X_train_inner, y_train_inner = X_train_outer[train_inner], y_train_outer[train_inner]
        X_val, y_val = X_train_outer[val], y_train_outer[val]
        
        # Standardize training and validation data
        scaler = StandardScaler()
        X_train_inner = scaler.fit_transform(X_train_inner)
        X_val = scaler.transform(X_val)
        
        # Fit ridge regression model for each lambda value and calculate validation error
        for k in range(lambdas.shape[0]):
            # Fit model on training data
            linearmodel = Ridge(alpha=lambdas[k])
            linearmodel.fit(X_train_inner, y_train_inner)
            
            # Calculate validation error
            y_val_pred = linearmodel.predict(X_val)
            errors[i, k, j] = mean_squared_error(y_val, y_val_pred)
    
    # Calculate mean validation error for each lambda value
    mean_errors = np.mean(errors[i, :, :], axis=1)
    
    # Find optimal lambda value
    optimal_lambda = lambdas[np.argmin(mean_errors)]
    
    # Train model with optimal lambda value on all training data
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)
    linearmodel = Ridge(alpha=optimal_lambda)
    linearmodel.fit(X_train_outer, y_train_outer)
    
    # Calculate test error
    y_test_pred = linearmodel.predict(X_test_outer)
    test_error = mean_squared_error(y_test_outer, y_test_pred)
    
    # Print optimal lambda value and test error for this outer CV fold
    print("Outer CV fold %d, Optimal lambda: %.2f, Test MSE: %.2f" % (i+1, optimal_lambda, test_error))

'''
#########################################################################################################
                   
###################### This section runs ONLY seperately from the others ###################################################################################
                          
                                # EVALUATION OF ANN WITH 2 LEVEL CV #


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


my_filename = '/Users/Nkely/dataR2.csv'
my_df = pd.read_csv(my_filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = my_df.values  


# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the 10 columns from inspecting 
# the file.
cols =[3, 5, 7, 8]
X = raw_data[:, cols]
y = raw_data[:, 4]
y = y.reshape(-1, 1)
N, M = X.shape

# Define the outer and inner folds
outer_kfold = KFold(n_splits=10, shuffle=True, random_state=42)
inner_kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the number of hidden units
hidden_units = [1, 2, 3, 4]

# Initialize variables
errors = np.zeros((outer_kfold.get_n_splits(), len(hidden_units), inner_kfold.get_n_splits()))

# Outer cross-validation loop
for i, (train_idx, test_idx) in enumerate(outer_kfold.split(X)):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # Inner cross-validation loop
    for j, (train_idx_inner, val_idx) in enumerate(inner_kfold.split(X_train_outer)):
        X_train_inner, X_val = X_train_outer[train_idx_inner], X_train_outer[val_idx]
        y_train_inner, y_val = y_train_outer[train_idx_inner], y_train_outer[val_idx]

        # Standardize training and validation data
        scaler = StandardScaler()
        X_train_inner = scaler.fit_transform(X_train_inner)
        X_val = scaler.transform(X_val)

        # Initialize variables
        best_h = None
        best_error = float('inf')

        # Train models for different numbers of hidden units
        for h in hidden_units:
            # Define model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X_train_inner.shape[1], h),
                torch.nn.Tanh(),
                torch.nn.Linear(h, 1),
            )

            # Train model
            net = model()
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters())
            for epoch in range(10000):
                inputs = torch.autograd.Variable(torch.Tensor(X_train_inner.astype(np.float32)))
                targets = torch.autograd.Variable(torch.Tensor(y_train_inner.reshape(-1,1).astype(np.float32)))
                optimizer.zero_grad()
                out = net(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()

            # Compute validation error
            X_val_tensor = torch.autograd.Variable(torch.Tensor(X_val.astype(np.float32)))
            y_val_pred = net(X_val_tensor).data.numpy()
            error = mean_squared_error(y_val, y_val_pred)

            # Check if this model is the best one so far
            if error < best_error:
                best_h = h
                best_error = error

        # Store the best validation error for this inner CV fold
        errors[i, j, :] = best_error

    # Calculate mean validation error for each number of hidden units
    mean_errors = np.mean(errors[i, :, :], axis=1)

    # Find optimal number of hidden units
    optimal_h = hidden_units[np.argmin(mean_errors)]

    # Standardize training and test data
    scaler = StandardScaler()
    X_train_outer = scaler.fit_transform(X_train_outer)
    X_test_outer = scaler.transform(X_test_outer)

    # Train model with optimal number of hidden units on all training data
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(X_train_outer.shape[1], optimal_h),
        torch.nn.Tanh(),
        torch.nn.Linear(optimal_h, 1),
    )

    net = model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    for epoch in range(10000):
        inputs = torch.autograd.Variable(torch.Tensor(X_train_outer.astype(np.float32)))
        targets = torch.autograd.Variable(torch.Tensor(y_train_outer.reshape(-1,1).astype(np.float32)))
        optimizer.zero_grad()
        out = net(inputs)
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

    # Compute test error
    X_test_outer_tensor = torch.autograd.Variable(torch.Tensor(X_test_outer.astype(np.float32)))
    y_test_outer_pred = net(X_test_outer_tensor).data.numpy()
    test_error = mean_squared_error(y_test_outer, y_test_outer_pred)

    # Print results
    print(f"Outer CV fold {i+1}: Optimal h={optimal_h}, Test error={test_error:.4f}")




'''
#########################################################################################################

                    ############    1.  LINEAR    vs      BASELINE  #################


import numpy as np, scipy.stats as st



# This script crates predictions from three KNN classifiers using cross-validation

test_proportion = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)


mA = linearmodel1
mB = baseline_model

yhatA = mA.predict(X_test)
yhatB = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings

# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - yhatA ) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_test - yhatB ) ** 2
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print("############    1.  BASELINE    vs    LINEAR  #############")
print ('Estimated difference:', np.mean(CI))
print ('Interval:', CI)
print ('pvalue:', p)



                    ############    2.  ANN    vs     BASELINE  #################

test_proportion = 0.2

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_proportion)

# Fit ANN model
ANN_model = ANN(X_train, y_train, h=3, max_iter=10000)

mA = baseline_model
mB = ANN_model

yhatA = mA.predict(X_test)
with torch.no_grad():
    yhatB = mB(torch.Tensor(X_test)).numpy()[:, np.newaxis]

print("############    2.  ANN    vs    BASELINE  #############")

# Compute mean squared errors
ANN_preds = ANN_model(torch.Tensor(X_test))
linear_preds = torch.Tensor(yhatA)
ANN_mse = np.mean((y_test - ANN_preds.detach().numpy()) ** 2)
linear_mse = np.mean((y_test - linear_preds.detach().numpy()) ** 2)

# Compute z with squared error for ANN model
zA = (y_test - ANN_preds.detach().numpy()) ** 2

# Compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute z with squared error for linear regression model
zB = (y_test - linear_preds.detach().numpy()) ** 2

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf(-np.abs(np.mean(z))/st.sem(z), df=len(z)-1)  # p-value

print("Mean difference:", np.mean(zA - zB))
print("Confidence interval:", CI)
print("p-value:", p)



                    ############    3.  ANN vs LINEAR  #################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as st

# Define ANN model
class ANN(nn.Module):
    def __init__(self, X, y, h=3):
        super(ANN, self).__init__()
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]
        self.hidden_layer = nn.Linear(self.input_dim, h)
        self.output_layer = nn.Linear(h, self.output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def train(self, X_train, y_train, max_iter=10000, learning_rate=0.01):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        for i in range(max_iter):
            optimizer.zero_grad()
            y_pred = self.forward(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()

# Split data into training and testing sets
test_proportion = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_proportion)

# Create ANN model
ANN_model = ANN(X_train, y_train, h=3)

# Train ANN model
ANN_model.train(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())

# Create linear regression model with L2 regularization (λ=1)
linear_model = nn.Linear(X_train.shape[1], y_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(linear_model.parameters(), lr=0.01, weight_decay=1) # Add weight decay parameter for L2 regularization
for i in range(10000):
    optimizer.zero_grad()
    y_pred = linear_model(torch.from_numpy(X_train).float())
    loss = criterion(y_pred, torch.from_numpy(y_train).float())
    loss.backward()
    optimizer.step()

# Evaluate models on testing data
ANN_preds = ANN_model(torch.from_numpy(X_test).float())
linear_preds = linear_model(torch.from_numpy(X_test).float())

# Compute mean squared errors
ANN_mse = np.mean((y_test - ANN_preds.detach().numpy()) ** 2)
reg_mse = np.mean((y_test - linear_preds.detach().numpy()) ** 2)

# Compute z with squared error for ANN model
zA = (y_test - ANN_preds.detach().numpy()) ** 2


# Compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute z with squared error for linear regression model
zB = (y_test - linear_preds.detach().numpy()) ** 2

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = 2*st.t.cdf(-np.abs(np.mean(z))/st.sem(z), df=len(z)-1)  # p-value

print("############    1.  ANN    vs    LINEAR  #############")
print("Estimated difference:", np.mean(zA - zB))
print("Interval:", CI)
print("pvalue:", p)





