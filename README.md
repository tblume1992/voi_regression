# voi_regression

What this script aims to do is to find a regression model which is "stable" in it's approximation for the coefficient of one of the variables.

To do this it first needs to know which variable you are interested in measuring, and what you believe is the effect this variable has on the y variable (In percentage terms).

Next, the script will measure how sensitive the coefficient is to model selection.  It does this by doing a random patches algorithm for a user defined number of iterations.  The results from this will be a distribution of the variable of interest's estimated coefficients derived from the randomly built models.  It may look something like this:

![alt text](https://github.com/tblume1992/voi_regression/blob/master/Capture.PNG)

This distribution will usually be a mixture distribution, where each sub-population is based on the inclusion or exclusion of one or many other variables.  To deal with this, the script will detect the number of distributions and ask for a visual verification.  If it all looks good then it will attempt to define each distribution based on the "average" model which makes up the distribution.  Currently, we define the average model based on a naive bayes with a probability for belonging to a sub population being greater than the:

median probability + 1 sd of the probabilities 

for that sub population.  If the visual verification looks incorrect then you should increase the number of iterations so the distribution detection has more data to work with, alternatively, tune the percentage of your dataset that is subsampled.

Finally, the script will take the "average" model which has the characteristics which is "closest" to the expected outcome defined by the user and run a GLM using MCMC sampling to build a posterior estimate for the probability distribution of the coefficient.
