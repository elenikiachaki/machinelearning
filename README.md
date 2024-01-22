## machinelearning
Machine learning project



# Project aim:
Our interest lies in predicting breast cancer based on anthropocentric data and parameters that can be gathered in routine blood analysis. There are 10 predictors, all quantitative, and a binary dependent variable, indicating the presence or absence of breast cancer. Prediction models based on these predictors, if accurate, can potentially be used as a biomarker of breast cancer and consequently allow for early detection ensuring a greater probability of having a good outcome in treatment.
More specifically, with the classification technique we will try to predict the condition of the individual(breast cancer or healthy) based on the metabolic and anthropocentric attributes. The attributes that will be used will be selected after applying an analysis that will determine which attributes out of the 9 have the maximal positive predictive value to the condition of the individual. As far as the regression task is concerned, even though the dataset is not commonly used for regression purposes, we could predict several attributes such as insulin based on the HOMA and/or glucose levels.

# Source of data
The UCI machine learning repository was used, from which we retrieved Breast Cancer Coimbra Data Set [1]. This data set came from 64 women newly diagnosed with breast cancer (BC) that were recruited from the Gynaecology Department of the University Hospital Centre of Coimbra (CHUC) between 2009 and 2013. All samples were naive, i.e. collected before surgery and treatment. On the other hand, the 52 controls were female healthy volunteers. All patients had had no prior cancer treatment and all participants were free from any infection or other acute diseases or comorbidities at the time of enrolment in the study.


# Instructions:
In order to run the python scripts of the repository, the dataset must be downloaded locally, and the paths to the file must be adjusted accordingly.
The scripts and the data set should be located in the same directory.


# Code Description:

Preprocessing.py:
Preprocessing of the data set: cleaning, summary statistics of the attributes, outliers idenification, investigation of corellation and covariance of attributes, visualization of attributes' distribution, PCA

Regression.py:
The present section includes the solution of a relevant regression problem for the Breast Cancer data set, as well as the statistical evaluation of the subsequent result.
Furthermore, three different Machine Learning models are here compared: the regularized linear regression model from the previous section, an artificial neural network (ANN) and a baseline in the regression problem defined previously. The aim is to investigate whether one model is better than the other or if either model performs better than a trivial baseline. In order to answer these questions, two-level cross-validation is applied, followed by statistical evaluation of the difference observed among the models’ performance.

Classification.py:
In this section the main binary classification problem is adressed: The prediction of breast cancer choosing the possible biomarkers. Three different methods are employed to do that, logistic regression (LR), decision trees (DT) and a baseline. Finally,  two level cross validation is utilized to evaluate their performances and they are compared using statistical testing.





# References: 

[1] https://archive.ics.uci.edu/dataset/451/breast+cancer+coimbra




