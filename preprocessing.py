#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:18:39 2023

@author: annamelidi
"""



import numpy as np
import pandas as pd
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, 
                               xticks, yticks,legend,show, suptitle, title)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load the csv data using the Pandas library
my_filename = '/Users/annamelidi/Documents/DTU_courses/2_semester/MachineL/Breast_Cancer_Data/dataR2.csv'
my_df = pd.read_csv(my_filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
my_raw_data = my_df.values  

# Notice that raw_data both contains the information we want to store in an array
# X1 and the information that we wish to store 
# in y1 (the class labels).

# We start by making the data matrix X1 by indexing into data.
# We know that the attributes are stored in the 9 columns from inspecting 
# the file.
my_cols = range(0, 9) 
X1 = my_raw_data[:, my_cols]
allmy_cols = range(0, 10) 

# We can extract the attribute names that came from the header of the csv
my_attributeNames = np.asarray(my_df.columns[my_cols])
allmy_attributeNames = np.asarray(my_df.columns[allmy_cols])

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
my_classLabels = my_raw_data[:,-1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
my_classNames = np.unique(my_classLabels)
# We can assign each type of class with a number by making a
# Python dictionary as so:
my_classDict = dict(zip(my_classNames,range(len(my_classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket.


# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y1 = np.array([my_classDict[cl] for cl in my_classLabels])

# We can determine the number of data objects and number of attributes using 
# the shape of X1
N1, M1 = X1.shape

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C1:
C1 = len(my_classNames)


print("Data were loaded.\n")


                   ######### Matrix Scatter Plot #########


figure(figsize=(12,10))

for m1 in range(M1):
    for m2 in range(M1):
        subplot(M1, M1, m1*M1 + m2 + 1)
        for c in range(C1):
            class_mask = (y1==c)
            plot(np.array(X1[class_mask,m2]), np.array(X1[class_mask,m1]), '.', alpha=.4, )
            
            if m1==M1-1:
                xlabel(my_attributeNames[m2])
            else:
                xticks([])
            if m2==0:
                ylabel(my_attributeNames[m1])
            else:
                yticks([])
            
legend(my_classNames)
suptitle('Matrix Scatter Plot',)

show()

print("The Matrix Scater Plot between all the pairs of attributes was generated.\n")

                           ##### Summary statistics #######



# Calculate mean for every attribute
mean = np.mean(my_raw_data, axis=0)

# Calculate variance for every attribute
variance = np.var(my_raw_data, axis=0)

# Calculate median for every attribute
median = np.median(my_raw_data, axis=0)


# Calculate range for every attribute
rng = np.ptp(my_raw_data, axis=0)


# Calculate standard deviation for every attribute
std_dev = np.std(my_raw_data, axis=0)


# Create a pandas DataFrame to store the summary statistics
summary_stats = pd.DataFrame({
    'mean': mean,
    'variance': variance,
    'median': median,
    'range': rng,
    'standard deviation': std_dev
}, columns=['mean', 'variance', 'median', 'range', 'standard deviation'], index=allmy_attributeNames)





                            ##### Covariance and Correlation #####
                            
# calculate the covariance matrix
df_cov = np.cov(my_raw_data, rowvar=False)

# calculate the correlation matrix
df_corr = np.corrcoef(my_raw_data, rowvar=False)

# Convert covariance matrix  and correlation matrix to pandas DataFrames
df_cov = pd.DataFrame(df_cov, columns=allmy_attributeNames)
df_cov.index = allmy_attributeNames

df_corr = pd.DataFrame(df_corr, columns=allmy_attributeNames)
df_corr.index = allmy_attributeNames

#Exporting dataframes to Excel file
summary_stats.to_excel("allsummary_stats.xlsx")
df_cov.to_excel("allcovariance.xlsx")
df_corr.to_excel("allcorrelation.xlsx")


print("Summary statistics are exported successfully to Excel Files.\n")



                            ##### Correlation Heatmap ######

# Create heatmap using seaborn library
sns.heatmap(df_corr, annot=True, cmap='coolwarm')

print("The correlation of the attributes is plotted on a heatmap.\n")





                    ###### Create a histogram for each column ########
                        


fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
axs = axs.ravel()

for i in range(X1.shape[1]):
    axs[i].hist(X1[:,i], bins=30)
    axs[i].set_title(my_attributeNames[i])
    mean = np.mean(X1[:,i])
    axs[i].axvline(mean, color='r', linestyle='dashed', linewidth=2)
    
plt.tight_layout()
plt.show()

print("Histogram of every attribute is generated.\n")







                               ##########   Normalization   ################

# Calculate the mean and standard deviation of each column
mean = np.mean(X1, axis=0)
std_dev = np.std(X1, axis=0)

# Subtract the mean from each element and divide by the standard deviation
normalized_X1 = (X1 - mean) / std_dev








                       ##################    PCA    #######################

 

# Subtract mean value from data
Y = normalized_X1 - np.ones((N1,1))*normalized_X1.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T  

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('Breast Cancer Coimbra Data Set : PCA')

for c in range(C1):
    # select indices belonging to class c:
    class_mask = y1==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(my_classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()




########## Compute variance explained by principal components   #############



rho = (S*S) / (S*S).sum() 
threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()



    
                #######Let's look at their coefficients ##################

pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M1+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, my_attributeNames, rotation=45 )            
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Breast Cancer: PCA Component Coefficients')
plt.show()


print("PCA was performed.\n")

               #### Create boxplots from the dataframe ######
                
                
plt.boxplot(normalized_X1)

# Add title and labels
plt.title("Boxplot of DataR2")
plt.xlabel("Attributes")
plt.ylabel("Values")

# Show the plot
plt.show()

print("Boxplots for the distrubution of every attribute are generated.\n")

                    ###### Parallel Coordinates Plot #######

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 5))

# Loop through each data point and plot the line
for i in range(normalized_X1.shape[0]):
    # If the class is 1, plot a blue line, otherwise plot a red line
    if my_classLabels[i] == 1:
        ax.plot(range(normalized_X1.shape[1]), normalized_X1[i], color='blue', alpha=.3)
    else:
        ax.plot(range(normalized_X1.shape[1]), normalized_X1[i], color='red', alpha=.3)

# Set the x-axis tick labels
ax.set_xticks(range(normalized_X1.shape[1]))
ax.set_xticklabels(my_attributeNames)

# Set the y-axis limits
ax.set_ylim([0, 7])


# Add a legend
ax.plot([], [], color='blue', label='Healthy')
ax.plot([], [], color='red', label='Breast Cancer')
ax.legend()

# Show the plot
plt.show()

print("Parallel Coordinates Plot is generated.\n")

