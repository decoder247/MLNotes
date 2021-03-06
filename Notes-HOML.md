# Hands-On Machine Learning (HOML) Notes

*Notes and summary from Aurélien Geron's HOML book*

#### ❗❗ Last stopped at page: 69 / 43         //   ~~*Catch-up @ [here](here_link)*~~

## Table of Contents

| Index | Section                                     |
|:----- | ------------------------------------------- |
| 1.    | [Preface summary](#sec1)                    |
| 2.    | [Chapter I-1: The ML Landscape](#sec2)      |
| 3.    | [Chapter I-1: Exercises](#sec3)             |
| 4.    | [Chapter I-2: End-to-End ML project](#sec4) |

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

* Problems requiring a lot of **hand-tuning and long list of rules**
* **Fluctuating environments**: ML adapts to new data
* **Complex** problems without a good solution at all
* **Getting insights** about complex problems

ML has the added advantage of:

* Being ***shorter, easier to mantain, and more accurate*** than manually learning about the problem, identifying patterns and architecturing a specific solution. I.e. take advantage of available computational power.

* Being ***adaptable to changes in new data.*** If the behaviour of new data is slightly different, the algorithm is able to adapt its solution without outside intervention. -> More independent than an analytical solution.

* Being able to ***solve problems that are either too complex***, or have no soluion. For complex solutions via traditional approaches - ML algorithms can be easily deployed, to solve complex problems, and find patterns in the problem space that is very hard to find traditionally.

* Finally, ML can help humans ***learn and discover patterns to the problem.*** I.e. reveal correlations. This is called Data Mining

### 2.3 Types of ML systems

#### 2.3.1 Main categories:

1. Supervision level: 
   
   A. Supervised
   B. Semisupervised
   C. Unsupervised
   D. Reinforcement learning

2. Ability to learn on the fly:
   
   A. Online
   B. Batch

3. Learning nature:
   
   A. Instance based (comparing new to known data points)
   B. Model-based learning (detect pattern in data for predictions)

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

* In ML, attribute means data type (e.g. mileage), feature means data type + value (mileage = 100)

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
  * **Common**: Association rule learning (Apriori, Eclat)
  * Visualisation algorithms (2D, 3D)
  * Clustering (K-means, DBSCAN, Hierarchical Cluster Analysis - HCA)
  * Anomaly detection and novelty detection (One-class SVM, Isolation forest)
  * Visualisation and dimensionality reduction (PCA, kernal PCA, Locally-linear embedding - LLE, t-distributed stochastic neighbour embedding - t-SNE)

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

#### 2.3.6 Batch learning / Offline learning

* Defn: Incapable of learning incrementally, uses ALL available data
* Also known as *offline learning* as it generally is computate/time-intensive, and thus done offline
* Although needs retraining for new data, this process can be automated!
* Applications: Non-volatile data. Not suitable for autonomous learning on limited resources (i.e. phones, rovers)

#### 2.3.7 Online learning / Incremental learning

* Defn: Learns incrementally in sequence, either individually or in small mini-batches. Also termed as *incremental learning*.

* Advantages:
  
  * On-the-fly learning
  * Suitable for applications w/ limited resources (phones, rovers)
  * Saves space by discarding trained data (unless want to roll back to previous state)
  * Suitable for training huge datasets beyond capacity of core's memory, i.e. *out-of-core learning*, typically done in mini-batches and done OFFLINE!

* Disadvantages:
  
  * Need to consider *learning rate*, i.e. how adaptable the system is to new data.
  * *Learning rate 🎗 Noise and Outliers sensitivity*. Having a high learning rate will 'place less significance' on the old data and lower inertia to noise
  * Close monitoring required when dependent on quality of new data. If the new data is unreliable, manual intervention might be required, or an *anomaly detection algorithm* for abnormal data detection.

#### 2.3.8 Instance based learning

* Having a good performance measure is not enough, *the key is to perform well in new instances, i.e.* ***generalisation*** .
* *Defn*: As seen below, the new instance will be classified as a triangle as most similar instance belong to that class!
  ![Instance-based learning][i1-15]
* Example: A system learns the examples by heart, and generalises new instances by comparing them to learned examples (or a subset of them), using a *similarity measure*.
* Types:
  * Similarity measure (Does not find an average of the 'nearest neighbours'. K-NN does this instead, which is a regression model)

#### 2.3.9 Model based learning

* *Defn*: As seen below a model is made instead for the prediction/classification of the new instance.
  ![Model-based learning][i1-16]

* Case study (not related to above image): Life satisfaction of a country can be modelled via a linear model, e.g. *selected* based on one *attribute*: GDP per capita. This selection step is called *model selection*.
  
  * life_satisfaction = θ<sub>0</sub> + θ<sub>1</sub> × GDP_per_capita
  * *Model parameters*: θ<sub>0</sub>, θ<sub>1</sub>

* To determine a parameter value, a *performance measure* is required!
  
  * *Cost function*: How bad the model is. Typically used in linear regression.
  * *Fitness / Utility function*: How good the model is
  * Example: In linear regression, a cost function is typically used to 'fit' the line to the data points, with a minimised cost/error between prediction and training examples.

* Once a performance measure is determined, e.g. the linear regression algorithm, it can be fed the data / applied to *train the model* and determine the parameter values!

* Case study revisited:
  
  * life_satisfaction = θ<sub>0</sub> + θ<sub>1</sub> × GDP_per_capita
  * *Model parameters*: θ<sub>0</sub>, θ<sub>1</sub>
  * Use *linear regression* to train model and determine parameter values!
  * Gather + study the data ➡
    Model selection ➡
    Train the model (Using the learning algorithm that employs a cost/fitness fnc and determines the model parameters) + tune parameters ➡
    Apply the model on new cases, *inference*, hoping it generalises well

* Types:
  
  * Linear regression
  * Polynomial regression
  * K-NN regression

### 2.4 Main Challenges of ML

* Insufficient Training Data

* Data quantity VS Algorithm quality
  
  * *Norvig's* 2009 paper on "The Unreasonable Effectiveness of Data" and *Banko/Brill*'s 2001 paper both suggest quantity of data yields more effective performance + larger gains, even with simpler algorithms.

* Nonrepresentative training data
  
  * Learning method needs to account for outliers so that it can accurately generalise
  * ❌ Too small sample: Sampling noise i.e. Nonrepresentative data as a result of chance
  * ❌ Too large sample: Possibility of flawed sampling, i.e. *sampling bias* where data is biased based on the sampling method/grouping.
    ![A more representative training sample][i1-21]

* Poor quality data
  
  * Might need pre-processing (Outliers, noise, errors, missing features)

* Irrelevant features
  
  * *Feature engineering*: Consideration of features in the training step. Useful features are important, otherise *garbage in, garbage out*!
    * *Feature selection*: Most useful features
    * *Feature extraction*: Combining existing features to produce smt more useful (dimensionality reduction algorithms!)
    * Creating new features

* Overfitting the training data
  
  * Overgeneralising ➡ i.e. works well on training data but not new data.
  * Overfitting happens when the model is too complex relative to noise, i.e. If the sample is too small, it will detect patterns in the noise itself
    ![Overfitting the training data][i1-22]
  * Solutions:
    * Simplify the model (linear > polynomial)
    * Gather more training data
    * Reduce noise
    * *Regularisation*: Constraining a model's parameters, i.e. by limiting its degrees of freedom
  * *Regularisation* can be controlled by a *hyperparameter*, which is a parameter of the learning algorithm (i.e. How a linear model and its parameters are determined) and not a model parameter itself (i.e. θ<sub>0</sub>, θ<sub>1</sub>). This is set prior to training, and remains constant during. Tuning hyperparameters is a challenge.
    * Too large: Flat model, not overfit training data
    * Too small: Overfitting might occur.

* Underfitting the training data
  
  * *Defn*: When a model is too simple to learn the underlying structure of the data
  * *Solutions*:
    * Selecting a more complex model, with more parameters
    * Feeding better features to the learning algorithm
    * Reducing the constraints of the model (e.g. reducing regularisation hyperparameter)

### 2.5 Testing and Validating

* To test, training data is typically split in to the *training set* (80%) and the *test set* (20%). 
* Error rate on new cases is called the *generalisation / out-of-sample error*, which evaluation on the test set will give a good estimation of.
* A low training error and high generalisation error indicates ➡ OVERFITTING!

#### 2.5.1 Hyperparameter tuning and model selection

* A challenge in hyperparameter tuning is tuning the hyperparameters too perfectly for *that particular test set*!
  * Example: In tuning a regularisation hyperparameter (to avoid overfitting) for a linear model, it is found hyperparameter 'A' produces a model with the least generalisation error of 5%. However in production it produces 15% errors!
* *Holdout Validation*: Solution to selecting/tuning/training a model that is not too adapted to the test set (to a T issue)
  * Segment a small part of the training set to a *validation set* or *dev set*. Multiple models with various hyperparameters are first trained on the smaller training set alone, and then the best model + hyperparameter is selected and trained on the full training set (including the *validation set*) to get the final model for later testing.
  * *Cross-validation*: Challenge here is determining the validation set size. Cross-validation solves this by using many small validation sets, and evaluating each model once per validation set (trained on REST of the data), and selecting the best one on average, before being trained on by the entire training set.
    ✔Better evaluation❌Longer training time

#### 2.5.2 Data Mismatch

* When applying to problems of the real world, nonrepresentative data for training, in relation to the new data, can be a real issue.
* Example: Image recognition app for plants on phones. Using web images to train is not very representative. Let's say we have limited representative data: 10 thousand camera imgs, and 10 million web imgs.
* MOST IMPORTANT RULE, validation and test set must be as representative as possible. Half test, half validation.
* If perform poorly on validation, a solution to find out if overfitting is an issue is by segmenting a training set further with a *train-dev set* (per Andrew Ng). After first training on the training set, the *train-dev set* is used for preliminary performance evaluation. A good evaluation performance determines overfitting to training data doesn't exist, and if it performs poorly on the validation set, data mismatch is the cause!
  * **Poor train-dev** set performance: Overfitting exists! ➡ Need to simplify / regularize model, get more data or clean up data as aforementioned.
  * **Good train-dev** set performance: Overfitting did not occur on training set, thus if poor validation set performance, Data Mismatch! ➡ Would need to make data more representative / accurate, i.e. via preprocessing.
* **No Free Lunch Theorem:** David Wolpert's 1996 paper asserts: *If no assumption is made about the data, there is no reason to prefer one model over another!*
  * There is no model that is *a priori* (from theory, reasoned truths) guaranteed to work better, some datasets fit a linear model better, others a neural network.
  * Since the ML approach to problem solving is not analytical, i.e. via a certain known theory about the data, there is no 'a priori' knowledge about the data, therefore no assumptions can be made! And because most problems are not purely mathematical, there is no strong *a priori* mathematical theory for it! Which is also why ML is preferred for these suitable problems.
  * **Clarification**:
    * *A priori* is defined as knowledge derived from logic, without the need for experience or sense data (i.e. rationalism), e.g. If x is red ➡ x has a color, If 5 > 3 ➡ 5 > 3!
    * *A posteriori* is defined as knowledge derived from sense experience / observations (i.e. empiricism), e.g. It frequently rains in the UK, Everest is the tallest mountain in the world.

---

## <a name="sec3"></a>3. Chapter I-1 Exercises

1. **ML Definition**?
   * An algorithm or method that is able to improve its performance with added input / data, typically for predictive or classification tasks.
2. **Four problem types where ML shines**?
   * Large amount of labelled data available
   * Too complex, i.e. no analytical solution
   * Require frequent updating of rules
   * Require insights into unfamiliar data
   * Volatile data environment
3. **Labelled training set definition**?
   * A training set with known, attached, answers.
4. **Two most common supervised tasks**?
   * Classification
   * Prediction
5. **Four common unsupervised tasks**?
   * Clustering
   * Visualisation and Dimensionality reduction
   * Anomaly and novelty detection
   * Association rule learning
6. **ML algo for allowing a robot to walk in unknown terrains**?
   * Reinforcement learning
7. **ML algo to segment customers into multiple groups?**
   * HCA - Heirarchal Cluster Algorithm
8. **Spam detection: Supervised or unsupervised learning?**
   * Supervised and semi-supervised, as the user can manually label spam
9. **Online learning system definition?**
   * Able to learn incrementally, but usually trained offline initially!
10. **Out-of-core learning definition?**
    * Learning beyond the memory capacity / compute-power of the computer being used for training. Usually occurs in context of online learning where the computer incrementally trains on mini-batches, before being trained on new instances. 
11. **Learning algo that relies on similarity measure?**
    * Instance-based learning learns training data by heart, then uses a similarity measure to classify / predict from existing instances.
12. **Model parameter VS learning algorithm's hyperparameter?**
    * A model parameter is the actual parameters to a model which needs to be determined during training, whereas a hyperparameter is a rule attached to that parameter, i.e. to regularise the parameter
13. **What do model-based learning algos search? Most common strategies? How predictions made?**
    * Model-based algos search for the model parameters using a performance measure. Most common performance measure include minimising a cost function. To make predictions, new instances are fed into the trained model.
14. **Main ML challenges?**
    * Lack of data, poor quality data, non-representative data, overfitting/underfitting training/testing data, data mismatch, uninformative features
15. **Model performs great one training data, but generalises poorly on new instances. What is happening? Name 3 solutions.**
    * Overfitted to training data. Simplify the model's complexity, i.e. decrease the order value. Remove data outliers. Make data as representative as possible. Increase data quantity.
16. **What is a test set and its use?**
    * Test the performance of a trained model.
17. **What use is a validation set?**
    * To compare models, and ensure that data mismatch is not occuring!
18. **Why don't you tune hyperparameters with the test set?**
    * Risk of overfitting the test set, with an optimistic generalisation error.
19. **Why repeated cross-validation > single validation set?**
    * Averaged performance result using slightly different training data, versus performance result using only 1 training data source. Extremely useful as it eliminates the need for a separate validation set.

---

---

## <a name="sec4"></a>4. Chapter I-2: End-to-End ML project*

**Pages 63-111*

### 4.1 Project overview

**Main steps in ML project:**

1. Look at the big picture

2. Get the data

3. Discover and visualise data to gain insights

4. Prepare the data for ML algorithms

5. Select a model and train it

6. Fine-tune your model

7. Present you solution

8. Launch, monitor and maintain your system

### 4.2 Datasets

**Open data repositories**

* UC Irvine Machine Learning Repository

* Kaggle Datasets

* Amazon's AWS Datasets

**Meta portals (for open data repo search) and others**

1. [dataportals.org](http://dataportals.org/)

2. [opendatamonitor.eu](http://opendatamonitor.eu/)

3. [quandl.com](http://quandl.com/)

4. [List of datasets for machine-learning research - Wikipedia](https://homl.info/9)

5. [Where can I find large datasets open to the public? - Quora](https://homl.info/10)

6. [Datasets subreddit](https://www.reddit.com/r/datasets)

## 4.2 Look at the big picture

<img title="" src="C:\Users\ivankl\AppData\Roaming\marktext\images\2020-05-18-23-07-42-image.png" alt="" width="677">

*Figure 2-1. 1990 California housing prices from StatLib repository*

Using the above dataset, we are provided with the population, median income, median housing price etc. for each block group or district. We will apply the main steps outlined in *Section 4.1* for this project.

### 4.2.1 Frame the problem

Building a model is not the end goal. **First question we need to ask - what is the exact business objective**? And how can the company expect to use and benefit from the model? This will determine:

1. What algorithms to select

2. Performance measure to use for model evaluation

3. Effort spent on tweaking model

Applying this to the example: we want a prediction of a district's median housing price for feeding to another ML system.

![](C:\Users\ivankl\AppData\Roaming\marktext\images\2020-05-18-23-23-48-image.png)

*Figure 2-2. A Machine Learning pipeline for real estate investments*

For this example, we want our model output to predict a district's median housing price to be fed to another ML system, which is fed other *signals*. As seen, our prediction will critically affect the downstream system.

> #### **Data Pipelines in ML**

> **Definition:** A *data pipeline* is a sequence of data processing *components*, typically run asynchronously (i.e. multiple threads).

> Pipeline are common in ML systems. Each component pulls a large amount of data, processes it, and outputs the result into a data store (i.e. one of the district data blocks in *Figure 2-2*), which is then retreived and used by the next component for processing. The **interface between components is thus these data stores**.
> 
> Advantages:
> 
> * Asynchronronous operations with data stores as interface between components.
> 
> * Each component is self-contained and generally a straightforward, simple design.  Different teams can focus on different components.
> 
> * Robust architecture - downstream components can continue to run normally for a while by using the last output of a broken-down component

> Disadvantages:

> * Require monitoring for broken-down components as data can get stale / lower performance if a broken component is unnoticed.

**Second question to ask - what does the current solution look like (if any)?** This will give a reference performance and insights into how to solve the problem.

For this example, the district (median) housing prices are currently manually estimated by experts using up-to-date information about a district, and complex rules. This is costly, time-consuming and usually inaccurate.

**We now need to frame the problem:** Is it supervised, unsupervised or reinforcement learning? Is it a classification task,  regression task, other? Batch or online?

For this example it is a supervised learning task given *labeled* training examples. This is a typical regression task since we are asked to predict a value - specifically a *multiple regression* problem since we will be **using multiple features** for a prediction. It is a *univariate regression* problem as we are **predicting a single value**. Moreover, there is no continuous flow of data coming in the system, so there is no particular need to adjust to changing data rapidly, plus datasize can fit in memory, so batch learning is suitable.

> Huge data can utlise from splitting batch learning work accross multiple servers (i.e. using *MapReduce* technique) or online learning.

### 4.2.3 Select a performance measure

A typical performance measure for regression problems is the *Root Mean Square Error* (RMSE). RMSE gives an indication to a system's prediction error and assigns a higher weight for large errors. 

$$
RMSE(X,h) = \sqrt{\frac{1}{m}.\sum_{i=1}^{m} (h(x^{(i)})-y^{(i)})^2}
$$

*Equation 2-1. Root Mean Square Error (RMSE)*

> #### Notations
> 
> $$



---

---

## <a name="secZ"></a>Z. References

[1]: https://www.coursera.org/learn/machine-learning/

[2]: https://www.coursera.org/course/neuralnets

[3]: http://shop.oreilly.com/product/0636920033400.do

[book-github]: https://github.com/ageron/handson-ml2 

[i1]: https://i.imgur.com/DhjVk61.png "source - https://imgur.com/a/qvyuhVh"

[i1-12]: ./Docs/HOML_fig1-12.png

[i1-15]: https://i.imgur.com/0tnEtR0.png "source-https://imgur.com/a/bXdpiOZ"

[i1-16]: https://i.imgur.com/9AacFOq.png "source - https://imgur.com/a/bR9Q2yR"

[i1-21]: https://i.imgur.com/qNcP7zE.png "source - https://imgur.com/a/EOlvpR5"

[i1-22]: https://i.imgur.com/1rbJLrs.png "source - https://imgur.com/a/RNmqcb8"
