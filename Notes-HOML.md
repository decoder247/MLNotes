# Hands-On Machine Learning (HOML) Notes
*Notes and summary from Aurélien Geron's book by the same name (HOML)*

#### ❗❗ Last stopped at page: 41 / 15.1

## Table of Contents
| Index | Section                                       |
|:-----:|-----------------------------------------------|
| 1.    | [Preface summary](#sec1)                      |
| 2.    | [Chapter I-1: The ML Landscape](#sec2)        |
| 3.    | [Chapter I-1: Exercises](#sec3)               |
| 4.    | [Chapter I-2: End-to-End ML project](#sec4)   |

---
---
## <a name="sec1"></a>1. Preface summary*
**Reference from p. 13-26*

### 1.1 Tools/Framework to be used:
* **Scikit-Learn (SL)**: FOSS ML framework
* **TensorFlow (TF)**: Google's ML framework
* **Keras (KR)**: High level ML API that can be used on top of Tensor (although tensor has its own implementation)
* **Python libs**: Numpy, Pandas, Matplotlib

### 1.2 PART I Roadmap - ML Fundementals w/ SL:
1. Understanding ML's concepts, categories and applications
2. Main steps in an ML project
3. Learning by fitting a model to data
4. Optimising a cost function
5. Handling, cleaning & preparing data
6. Selecting & engineering features
7. Cross-validation: model selection and hyperparameters tuning
8. Underfitting and overfitting challenges (bias/variance tradeoff)
9. Curse of dimensionality (Dimensionality reduction)
10. Unsupervised: Clustering, density estimation and anomaly detection
11. Common: Linear/Polynomial regression, logistic regression, KNN, Support Vector Machines, Decision Trees, Randome forests and Ensemble methods 

### 1.3 Part II Roadmap - Neural Nets (NNs) and Deep Learning (DL) w/ TF + KR:
1. Intro to Neural Nets (applications and definition)
2. Building and training NNs using Tensorflow + Keras
3. Architecture: Feedforward NNs, CNNs, Recurrent nets, LSTM Nets, autoencoders and GANs
4. Deep NNs training techniques
5. Scaling for large datasets (for NNs)
6. Reinforcement learning
7. Handling uncertainty with Bayesian Deep Learning

### 1.4 Useful Resources (Further reading // Code examples):
* Andrew Ng's course on [ML in coursera][1]
* Geoffrey Hinton's course on [NNs and DL][2]
* Joel Grus' [Data Science from scratch][3]
    * Hard-code from scratch
* François Chollet *Deep Learning with Python*
    * Creator of Keras libary. Very practical approach.
* Mostafa, Ismail, Lin *Learning from Data*
    * More theoratical approach
* Code examples used within this book: [GitHub Link][book-github]

---
## <a name="sec2"></a>2. Chapter I-1: The ML Landscape
**Reference from p. 29 - 61*

### 2.1 ML Definition
A computer program is said to learn from experience (E), with respect to task (T) and some performance measure (P), if its performance on T, as measured by P, improves with experience E. 
> I.O.W. - A program with experience, should perform better on new tasks if it is said to be 'learning'.

### 2.2 The Traditional approach vs the ML approach
![The traditional approach][i1]
Traditional, analytical approaches require an iterative process where you look at a set of problem, identify and devise a set a solutions that tackle a subset of the problem, and iterate from there so that the final solution solves the problem completely - i.e. a normal software dev process.

To summarize, ML is great for:
* Problems requiring a lot of hand-tuning and long list of rules
* Fluctuating environments: ML adapts to new data
* Complex problems without a good solution at all
* Getting insights about complex problems

However, instead of learning about the problem ourselves, identify patterns to solve ourselves, and specific solutions to each problems ourselves, we can get a computer to do it, faster and more straightforwardly with all the 21st century computational power made available to us. ***This is usually shorter, easier to mantain, and more accurate.***

In addition to this, the ML approach is adaptable to changes in new data. I.e. ***if the behaviour of new data is slightly different, the algorithm is able to adapt its solution without outside intervention. -> More independent than an analytical solution.***

Further, ML can solve problems that are either too complex. For complex solutions via traditional approaches - ***ML algorithms can be easily deployed, to solve complex problems, and find patterns in the problem space that is very hard to find traditionally***.

Finally, ML can ***help humans learn and discover patterns to the problem. I.e. reveal correlations. This is called Data Mining***

### 2.3 Types of ML systems
#### 2.3.1 Main categories:
1. Supervision level (A. Supervised, B. Semisupervised, C. Unsupervised, D. Reinforcement learning)
2. Ability to learn on the fly (A. Online vs B. Batch)
3. A. Instance based (comparing new to known data points) vs B. Model-based learning (detect pattern in data for predictions)

#### 2.3.2 Supervised Learning:
* *Classification*: Feed data with labels
* *Prediction via regression*: Feed predictors (features), to predict a task's future values
* Regression algorithms can be used for classification, i.e. *logistic regression* (probability of belonging to a class)
* Types:
    * KNN
    * Regressions (Linear, logistic)
    * Support vector machines (SVM)
    * Decision trees and random forests
    * NNs (not all are supervised!)

Note:
* Attribute is data type (e.g. mileage), feature is data type + value (mileage = 100)

#### 2.3.3 Unsupervised learning:
* Unlabelled learning, i.e. without an answer set
* *Clustering*: Helps segment/group data points. A *HCA* is able to further subdivide each group into smaller groups.
* *Visualisation*: Represented in 2D/3D space to manually inspect for data. Some smart clustering can be incorporated to prevent overlaps.
* *Dimensionality reduction*: Simplify data without losing too much info (i.e. compression)
    * i.e. feature extraction: Using a car's age to represent the wear and tear of a car
    * Usually done prior to a supervised learning method to simplify the problem set
* *Anomaly detection*: Removing outliers before feeding into learning method. Tolerant even with small % of outliers in training set.
* *Novelty detection*: Similar to anomaly, but less tolerant and only expects normal data during training
* *Association rule learning*: Discovers interesting relations between attributes (i.e. Beef shoppers also buy chicken)
* Types:
    * Visualisation algorithms (2D, 3D)
    * Clustering (K-means, DBSCAN, Hierarchical Cluster Analysis - HCA)
    * Anomaly detection and novelty detection (One-class SVM, Isolation forest)
    * Visualisation and dimensionality reduction (PCA, kernal PCA, Locally-linear embedding - LLE, t-distributed stochastic neighbour embedding - t-SNE)
    * Association rule learning (Apriori, Eclat)

#### 2.3.4 Semisupervised learning:
* Partially labelled data, i.e. a lot of unlabelled, and a bit of labelled data
* E.g. Clustering an identified person in a set of photographs, once a label is given, the person is known across the photographs. Usually there are several clusters for the same person, repeated labelling improves accuracy.
* Types:
    * Deep Belief Networks (DBN)
        * Comprised of stacked Restricted Boltzmann Machines 
        * RBMs trained sequentially, unsupervised, and fine-tuned using supervised techniques
    * Unsupervised pretraining

#### 2.3.5 Reinforcement Learning
* Learning system based on *Agents* - observe the environment, select and perform actions, get *rewards/penalties* in return.
* Learn the best *policy* (strategy) for most points via rewards/penalties for the *agent* to take
* Application: Games, Robots (Tactic: Playing against itself)

![Reinforcement Learning][i1-12]

#### 2.3.6 Batch and online learning
* *Batch Learning*: 

---
## <a name="sec3"></a>3. Chapter I-1 Exercises
1. How would you define ML?
* An algorithm that is able to improve its performance, given a dataset, typically tasked with classification or predictive tasks
2. Name four problems where it shines?
* 

---
---
## <a name="sec4"></a>4. Chapter I-2: End-to-End ML project
**Pages 63-111*

---
---
## <a name="secZ"></a>Z. References
[](Reference!------)  

[2]: https://www.coursera.org/course/neuralnets
[3]: http://shop.oreilly.com/product/0636920033400.do
[book-github]: https://github.com/ageron/handson-ml2 
[i1-12]: \Refs\HOML_fig1-12.png
[i1]: https://i.imgur.com/DhjVk61.png "source - https://imgur.com/a/qvyuhVh"