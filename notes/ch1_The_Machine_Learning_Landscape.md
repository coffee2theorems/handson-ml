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

'''
	import matplotlib
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import sklearn

	#Load the data
   	oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
	gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',',
