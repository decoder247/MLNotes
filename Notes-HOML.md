# Hands-On Machine Learning (HOML) Notes
*Notes and summary from Aurélien Geron's book by the same name (HOML)*

#### ❗❗ Last stopped at page: 38

## Table of Contents
| Index | Section                                       |
|:-----:|-----------------------------------------------|
| 1.    | [Preface summary](#sec1)                      |
| 2.    | [Chapter I-1: The ML Landscape](#sec2)        |
| 3.    | [Chapter I-1: Exercises](#sec3)               |
| 4.    | [4. Chapter I-2: End-to-End ML project](#sec4)|

---
## <a name="sec1"></a>1. Preface summary*
**Reference from p. 13-26*

#### Tools/Framework to be used:
* **Scikit-Learn (SL)**: FOSS ML framework
* **TensorFlow (TF)**: Google's ML framework
* **Keras (KR)**: High level ML API that can be used on top of Tensor (although tensor has its own implementation)
* **Python libs**: Numpy, Pandas, Matplotlib

#### PART I Roadmap - ML Fundementals w/ SL:
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

#### Part II Roadmap - Neural Nets (NNs) and Deep Learning (DL) w/ TF + KR:
1. Intro to Neural Nets (applications and definition)
2. Building and training NNs using Tensorflow + Keras
3. Architecture: Feedforward NNs, CNNs, Recurrent nets, LSTM Nets, autoencoders and GANs
4. Deep NNs training techniques
5. Scaling for large datasets (for NNs)
6. Reinforcement learning
7. Handling uncertainty with Bayesian Deep Learning

#### Useful Resources (Further reading + Code examples):
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

#### Definition
A computer program is said to learn from experience (E), with respect to task (T) and some performance measure (P), if its performance on T, as measured by P, improves with experience E. 
> I.O.W. - A program with experience, should perform better on new tasks if it is said to be 'learning'.

#### The traditional approach VS the new - ML
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

#### Types of ML systems
Main categories:
1. Supervision level (Sup, semi, un, reinforcement)
2. Ability to learn on the fly (Online vs Batch)
3. Instance based (comparing new to known data points) vs model-based learning (detect pattern in data for predictions)

Supervised Learning:
* *Classification*: Feed data with labels
* *Prediction via regression*: Feed predictors (features), to predict a task's future values
* Regression algorithms can be used for classification, i.e. *logistic regression* (probability of belonging to a class)
* Types:
    * KNN
    * Regressions (Linear, logistic)
    * Support vector machines (SVM)
    * Decision trees and random forests
    * NNs (not all are supervised!)

Unsupervised learning:
* Unlabelled learning, i.e. without an answer set
* *Clustering*: Helps segment/group data points. A *HCA* is able to further subdivide each group into smaller groups.
* *Visualisation*: Represented in 2D/3D space to manually inspect for data. Some smart clustering can be incorporated to prevent overlaps.
* *Dimensionality reduction*: Simplify data without losing too much 
* Types:
    * Visualisation algorithms (2D, 3D)
    * Clustering (K-means, DBSCAN, Hierarchical Cluster Analysis - HCA)
    * Anomaly detection and novelty detection (One-class SVM, Isolation forest)
    * Visualisation and dimensionality reduction (PCA, kernal PCA, Locally-linear embedding - LLE, t-distributed stochastic neighbour embedding - t-SNE)
    * Association rule learning (Apriori, Eclat)

Semisupervised learning:
* Types:
    * 
    * Unsupervised pretraining

Note:
* Attribute is data type (e.g. mileage), feature is data type + value (mileage = 100)


## <a name="sec3"></a>3. Chapter I-1 Exercises
1. How would you define ML?
* An algorithm that is able to improve its performance, given a dataset, typically tasked with classification or predictive tasks
2. Name four problems where it shines?
* 

## <a name="sec4"></a>4. Chapter I-2: End-to-End ML project
**Pages 63-111*

---
## <a name="secZ"></a>Z. References
[](Reference!------)  

[2]: https://www.coursera.org/course/neuralnets
[3]: http://shop.oreilly.com/product/0636920033400.do
[book-github]: https://github.com/ageron/handson-ml2 
[i1]: https://i.imgur.com/DhjVk61.png "source - https://imgur.com/a/qvyuhVh"