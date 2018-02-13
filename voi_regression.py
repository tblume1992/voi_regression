# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:07:26 2017

@author: TBlume
"""
print ('Loading Packages')
from sklearn.naive_bayes import GaussianNB
import pymc3 as pm
from theano import *
from sklearn import linear_model
import pandas as pd
import numpy as np
from random import randint
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import mixture
###############################################################################
###############################################################################
###############################################################################
'''USER INPUT PARAMETERS'''
iterations = 5000
# Expected effect (in percentage) that the variable of interest has on the y variable
expected_outcome = 0
y_variable = 'variable name'
variable_of_interest = 'variable name'
mcmc_iterations = 5000
#Percentage of your dataset you want to subsample for each random model
bootstrap_percentage = .3
# The number of bins created for the histogram
hist_bins = 100
"""Sales Data"""
df = pd.read_csv(FILEPATH)
###############################################################################
###############################################################################
###############################################################################


dataset = df
print("Does your y variable contain 0's?")
print('[y/n]')
user_input = input()
if user_input == 'y':
    dataset[y_variable] += 1
look = dataset[y_variable]
dataset[y_variable] = np.log(df[y_variable]) 
dataset = dataset.dropna()
train = dataset
dataset = dataset.drop(y_variable, 1)
#Build dataset which we will add the coefficients to
coef_index_vals = list(dataset)
coef_index_vals = pd.DataFrame(coef_index_vals).set_index([0], drop = True)
count = 1
coefficients = pd.DataFrame(index = coef_index_vals.index)
#Random patches loop with progress bar
mini_batch_count = 1
mini_batch_coef = pd.DataFrame(index = coef_index_vals.index)
print('Conducting random patches model')
for i in tqdm(range(iterations)):
    bootstrapped = train.sample(frac=bootstrap_percentage, replace=True, weights=None, random_state=None, axis=None)
    rand = randint(10, len(coef_index_vals))
    y = bootstrapped[y_variable]
    X = bootstrapped.drop(y_variable,1)
    VOI = X[variable_of_interest]
    X = X.drop(variable_of_interest)
    random_subspaces_X = X.sample(n=rand, replace=False, weights=None, random_state=None, axis=1)
    regent  = linear_model.LinearRegression()
    random_subspaces_X[variable_of_interest] = pd.Series(VOI)
    index_vals = list(random_subspaces_X)
    index_vals = pd.DataFrame(index_vals).set_index([0],drop = True)
    regent.fit(random_subspaces_X,y.values.ravel())
    coef = regent.coef_
    coef = pd.DataFrame(coef)
    coef.columns = [count]
    coef.index = index_vals.index
    mini_batch_coef = mini_batch_coef.join(coef, how='left', lsuffix='_left')
    if mini_batch_count == 200:
        coefficients = coefficients.join(mini_batch_coef, how='left', lsuffix='_left')
        mini_batch_coef = pd.DataFrame(index = coef_index_vals.index)
        mini_batch_count = 1
    count = count + 1
    mini_batch_count += 1
coefficients = coefficients.join(mini_batch_coef, how='left', lsuffix='_left')    
se = coefficients.std(axis = 1, skipna = True) 
se = pd.DataFrame(se)
se = (se) 
beta = coefficients.mean(axis = 1, skipna = True)
beta = pd.DataFrame(beta)
t_test = beta/se
coefficients = coefficients.transpose()
coefficient = coefficients.fillna(1)
coefficient = np.where(coefficient != 1, 0, 1)
coefficient = pd.DataFrame(coefficient)
coefficient_y = coefficients[variable_of_interest]
columns = list(coefficients)
coefficient.columns = columns
coefficient = coefficient.drop(variable_of_interest, 1)
columns = list(coefficient)
coefficient = [coefficient_y, coefficient]
coefficient = np.column_stack(coefficient)
coefficient = pd.DataFrame(coefficient)
coefficient = coefficient.dropna()
coefficient_y = coefficient[0]
coefficient = coefficient.drop([0], 1)
coefficient.columns = columns
regent  = linear_model.LinearRegression()
regent.fit(coefficient,coefficient_y.values.ravel())
coef = regent.coef_
coef = pd.DataFrame(coef)
coef.index = columns

"""
http://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_nw_panel.html?highlight=panel#statsmodels.stats.sandwich_covariance.cov_nw_panel
statsmodels.stats.sandwich_covariance.cov_nw_panel(results, nlags, groupidx, weights_func=<function weights_bartlett>, use_correction='hac')
"""

variable_ofinterest = coefficients[variable_of_interest].dropna()
cluster_count = 1
cluster_criterion = pd.DataFrame([])
#Clustering to find the optimal number of distributions based on the BIC
#Max number of clusters we can idnetify is 15 although it is unlikely it will ever be that high unless you have high dimensions
#If it does look incorrect then just increases the number of random patches iterations
while cluster_count < 15:
    clf = mixture.GaussianMixture(n_components=cluster_count, covariance_type='full')
    clf.fit(variable_ofinterest.values.reshape(-1, 1))
    bics = clf.bic(variable_ofinterest.values.reshape(-1, 1))
    cluster_criterion = cluster_criterion.append([bics],[cluster_count])
    cluster_count += 1

#Re-cluster with the optimal number of clusters
cluster_criterion = cluster_criterion.idxmin()
number_of_clusters = cluster_criterion.at[0] + 1
clf = mixture.GaussianMixture(n_components=number_of_clusters, covariance_type='full')
clf.fit(variable_ofinterest.values.reshape(-1, 1))
predictions = clf.predict(variable_ofinterest.values.reshape(-1, 1))
variable_split = pd.DataFrame(np.column_stack([variable_ofinterest,predictions]))
variable_split.columns = ['Coefficient', 'Cluster']
d = {}
#Get Means of the clusters
distribution_means = variable_split.groupby(['Cluster'])[['Coefficient']].mean()

nearest_model = (distribution_means - expected_outcome)**2
nearest_model = nearest_model.idxmin()
nearest_model = nearest_model.at['Coefficient']
#Plot histograms with centers of mass and check to ensure it was done well
plt.hist(coefficients[variable_of_interest], bins=hist_bins, color='c')
x1 = distribution_means.iloc[[0]].values
count = 0
print ('***Visual Check***')
print ('We have found ' + str(number_of_clusters) + ' distributions!')

while count < len(distribution_means):
    plt.axvline(distribution_means.iloc[[count]].values, color='b', linestyle='dashed', linewidth=2)
    count += 1


plt.show()
print ('Does this look correct?')
print('[y/n]')
user_input = input()
#Continue if clustering looks correct
if user_input == 'y':
    for name, group in variable_split.groupby(['Cluster']):
        d['group_' + str(name)] = group
    '''
    for key, value in d.items():
        print('Distribution'+ str(key))
        pd.DataFrame.hist(d[key], column = 'Coefficient', bins = 40)
        plt.show()
    '''
    coefficients = coefficients.drop([variable_of_interest], 1)
    coefficient = coefficients.fillna(0)
    coefficient = np.where(coefficient != 0, 1, 0)
    coefficient = pd.DataFrame(coefficient)
    coefficient_y = variable_split['Cluster']
    columns = list(coefficients)
    coefficient.columns = columns
    coefficient = [coefficient_y, coefficient]
    coefficient = np.column_stack(coefficient)
    coefficient = pd.DataFrame(coefficient)
    coefficient = coefficient.dropna()
    coefficient_y = coefficient[0]
    coefficient = coefficient.drop([0], 1)
    coefficient.columns = columns
    regent  = GaussianNB()
    regent.fit(coefficient,coefficient_y.values.ravel())
    variable_probabilities = regent.theta_
    y_pred = regent.predict(coefficient)
    median_probability = np.median(variable_probabilities, axis = 1)
    std_probability = np.std(variable_probabilities,axis = 1)
    variable_probabilities = pd.DataFrame(variable_probabilities)
    variable_probabilities = variable_probabilities.transpose()
    variable_probabilities = variable_probabilities[~(variable_probabilities< (median_probability-2*std_probability))]  
    variable_probabilities = variable_probabilities.transpose()
    true_std_prob = np.std(variable_probabilities,axis = 1)
    variable_probabilities = pd.DataFrame(variable_probabilities)
    variable_probabilities = variable_probabilities.transpose()
    Min_model = pd.DataFrame(variable_probabilities[variable_probabilities > (median_probability + true_std_prob)])
    Min_model['Variables'] = list(coefficient)
    Min_model = Min_model.set_index('Variables', drop = True).fillna(0)
    min_model = Min_model[[nearest_model]]
    min_model = min_model[min_model > 0].dropna()
    min_model = min_model.index.tolist()
    
    min_model.append(variable_of_interest)
    print('We have built the stable model controlling for:' + str(min_model))
    print ('Does this look correct?')
    print('[y/n]')
    user_input = input()
    min_model.append(y_variable)
    if user_input == 'y':
        Structural_dataset = train[min_model]
        y = Structural_dataset[[y_variable]].values
        structural_dataset = Structural_dataset.drop(y_variable,1)
        x = structural_dataset
        data = dict(x=x, y=y)
        with pm.Model() as model:
            pm.glm.GLM.from_formula('y ~ x', data)
            trace = pm.sample(mcmc_iterations, cores=2)
        plt.figure(figsize=(7, 7))
        pm.traceplot(trace[100:])
        plt.tight_layout();
    
        pm.summary(trace)
