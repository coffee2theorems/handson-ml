# The Machine Learning Landscape

## Problems that Machine learning is great for
    * Problems that have lots of rules or long complicated algorithms 
    * Complex problems that have no good solution
    * Fluctuating environments
    * Getting insights about complex problems and large amounts of data

## Types of Machine Learning

There are four major cateogories:
1. Supervised Learning
2. Unsupervised Learning
3. Semisupervised Learning
4. Reinforcement Learnig


#### Supervised Learning
Training data has the desired output, called *labels* 

Typical supervised learning 
    ** classification (example Spam Filter) 
    * Regression
        -  predicting a *target* number (such as the price of a car) given a set of features (mileage, age, brand) called *predictors*
        
Common supervised learning algorithms 

* k-Nearest Neighbors
* Linear Regression
* Logistic Regression
* Support Vector Machines (SVMs)
* Decision Trees adn Random Forests
* Neural Networks


#### Unsupervised Learning 
Training data is unlabeled (System tries to learn without a teacher)

Common unsupervised Algorithms 

* Clustering
    * k-Means
    * Hierarchical Cluster Analysis (HCA)
    * Expectation Maximization
* Visualization and dimensionality reduction
    * Principal Component Analysis (PCA)
    * Kernel PCA
    * Locally - Linear Embedding (LLE)
    * t-distriubtion Stochastic Neighbor Embedding (t-SNE)
* Association rule learning
    * Apriori
    * Eclat

**Dimensionality Reduction** 
The goal of dimensionality reduction is to simplfy the data set, with out loosing too much data. 
To do this we merge correlated data together. By reducing our data, it takes less space on disk, 
and runs through our algorithm faster.

**Anomaly detection**
This is automatically removing outliers from a dataset before feeding it to another learning algortihm. 
We train the system on normal instances so that when it sees a new instance it can tell whether it looks like a normal one or whether it is likely a anomly 

**Association rule learning**
The goal of Association rule learning is to dig into a large data set and discover interesting relations between attributes. 


#### Semisupervised Learning
When an algorithm can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. 
*Example: Google photos notices the same people in different photos and then asks you who each person is.*

Deep belief networks (DBN) are based on unsupervised componenets of restricted Boltzmann Machines (RBMs) stacked ontop of one another. 
They are trained sequentially in an unsupervised manner and then the whole system is fine tuned with supervised learning techniques. 


#### Reinforcement Learning
The *agent* (Learning System) observes the enviroment and can select and perform actions to get *rewards* (or *penalties*) 
It comes up with the best *policy* (or strategy) 

*Examples: Deepminds Alpha Go, which learned how to play Go. And Robots that learn how to walk*


## Batch vs Online Learning

#### Batch Learning
The system can not learn incrementally, it is trained on all the available data. 
* Generally takes a lot of time and computing rescources
* Typcially done offline
* Once trained it is launched into production and does not need to be trained anymore. 
Often called **offline learning**

* If you want a system to learn about new data you must completely retrain the system with the new (and old) data. 
    * This can be automated
* Not used for systems that need to react rapidly (Stock exchange)
* Can cost a lot of time / computer resources to train on large data sets over and over.
* Not good for low power devices like phones with limited storage.

#### Online Learning
The system is trained incrementally by feedint it data instances sequentially, either individually or by small groups called *mini-batches*
* Great for systems that recieve a constant flow of data (Stocks) 
* Can change rapidly
* Once it has learned using a data set it can disregard it
    * Uses less storage for data
* Online learning algorithms can also be used to train systems on huge datasets that cannot fit in one machines main memory (*out-of-core learning*)
* *learning rate* is how fast the system can adapt to new data
    * High learning rate then you system will rapidly adapt to new data but quickly forget old data
    * Low learning rate has more "inertia" ; learns slowly but is less sensitive to noise in the new data or to sequences of nonrepresentative data
* If bad data is fed into the system its performance will slowly decline
    * May want to monitor incoming data using anomaly detection algorithms


## Instance Based vs Model Based Learning
How machine learning (ML) systems generalize 

#### Instance Based Learning
The system learns the examples by heart and then generalized to new cases using a *similarity measure* 

#### Model Based Learning
Build a model of examples then use the model to make *predictions*

* We use parameters (often written as theta) to control our models
* We then measure out models using *fitness fucntions* (or *utility fucntions*) that tell us how good our model is
    * Conversely a *cost function* tells us how bad our model is
	* People often use a cost function on linear models to tell them how far their model is from the data
	  The objective is to minimize the cost (distance between model and data)


Summary
* Study the data
* Selected a model
* Train the model on the training data (the learning algorithm searched for the model parameter values that minimize a cost function)
* Apply the model to make a predicition (called *inference*) 

Example code for a linear model


	import matplotlib
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import sklearn

	#Load the data
   	oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
	gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',',delimiter='\t',
					encoding='latin1', na_values="n\a")
	
	#Prepare the data
	country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
	X = np.c_[country_stats["GDP per Capita"]]
	y = np.c_[country_stats["Life satisfaction"]]
	
	# Visulaize the data
	country_stats.plot(kind="scatter", x="GDP per capita", y='Life satisfaction')
	plt.show()

	# Select a linear model
	model = sklearn.linear_model.LinearRegression()

	
	# Optional k-Nearist Neighbor
	# model = sklearn.neighbors.KNeighborRegressor(n_neighbors = 3)


	# Train the model
	model.fit(X, y)

	# Make a predicition for Cyprus
	X_new = [[22587]]  # Cyprus' GDP per capita
	print(model.predict(X_new)) ## outputs [[ 5.96242338]]


## Main Challenges of Machine Learning
Two main issues: "Bad Algorithm" or "Bad Data"

#### Data
Sometimes we don't have enough data to train an algorithm effeciently 
In the paper 'The unreasonable effectiveness of Data" researches showed that given large data sets simple algorithms can perform just as well as complicated ones.
Comming up with enough data can be costly though.

#### Non representative Data
In order to generalize data it is crucial that our training data represents the new cases we want to generalize. 
* This matters in both instances based and model based learning

If a sample set is too small we can get *sampling noise* (i.e. nonrepresentative data as a result of chance)

Even if a sample set is very large samples can still be nonrepresentative because the sampling method is flawed (*sampling bias*)

#### Poor Quality Data
Data can often contain errors, outliers, and noise (due to poor quality measurments) 

#### Irrelevant Features
*Feature Engineering* involves the following:
* Feature Selection: selecting the most useful features to train on among exisitng featrues
* Feature Extraction: combining exisitng features to produce a more  useful one (dimensionality reduction for example)
* Creating new features by gathering new data

#### Overfitting the Training Data
*Overfitting* is when the model performs well on the training data but does not generalize well.

Complex models such as deep neural networks can detect subtle patterens but if there isn't enough data (hence introducing sampling noise) the model is likely to detect patterens in the noise

Overfitting happens when the model is too complex relative to the amount and noisiness of the training data. Possible solutions are:
* To simplify the model, one with fewer parameters 
* Gather more data
* Reduce the noise in the training data (fix data errors and remove outliers)

*regularization* is when we constrain a model to make it simpler  and reduce the risk of overfitting.
* The amount of regularization to apply to a model is controlled by a *hyperparameter*
    * A hyperparameter is controlled by the learning algorithm not the model. It is set before the training
    * A large hyperparameter will cause the slope to close to zero. This will prevent overfitting but the model won't be as accurate.

*Degrees of Freedom* refers to the number of parameters we allow our model to change. (i.e. our linear model has 2, the slope and y intercept)

#### Underfitting the Training Data
*Underfitting* occurs when the model is too simple to learn the underlying structure of the data. 

To fix underfitting
    * select a more powerful model (i.e. more parameters)
    * feeding better features to the learning algorithm (feature engineering) 
    * reduce teh constraints on the model (reducing the regularization hyperparameter)


## Stepping Back 
* Machine Learing is about making machines get better at some task by learning from dat instead of having ot explicitly code rules.
* There are many different types of ML systems: supervised or not, Batch or online, instance-based or model-based, and so on.
* ML Project: Gather Training Data --> Feed the Training set to the learning algorithm 
    * If the algorithm is model based it tunes some parameters to fit the model to the training set
    * If the algorithm is instance based it learns the examples by heart and uses a similarity measure to generalize to new instances. 
* The system will not perform well if the training set is too small, or if the data is not representative, noisy, or polluted with irrelevant features. 
    * Model needs to be neither too simple (underfit) or too complex (overfit)

## Testing and Validating
Once you have created a model we will need to test to see if it works

One way to do this is to watch and monitor how well it performs. (Works okay if your model is good. Is horrible if your model is bad)

Better way to do it is to split your training data in half, the *Training set* and the *Test set*
* Train on the training set
* Test on the testing set

*Generalization error* is the error rate on teh new cases (also know as *out-of-sample error*) This value will tell you how your model will performon instances it has never seen

If the training error is low (i.e. your model makes a few mistakes on the training set) but the generalization error is high, that means your model is overfitting the training data.

If you are needing to decide between two models you can train both and compare how well they generalize.

*validation set* Train multiple models with various parameters using the training set, you select the model and hyperparameters tha perfom best on the validation  set,
 and when you're happy with your model you run a single final test against the test set to get an estimate of the geralization error. 

To avoid wasting too much training data in validation sets use *cross-validation* the training set is split into complementary subsets, and each model is trained against a different combination of these
subsets and validated against the remaining parts. 

Once the model type and hyperparameters have been selected the final model is t rained usig the full training set. 

