# Hands-On Machine Learning (HOML) Notes
*Notes and summary from Aurélien Geron's book by the same name (HOML)*

#### ❗❗ Last stopped at page: 32

## Table of Contents
| Index | Section                                |
|:-----:|----------------------------------------|
| 1.    | [Preface summary](#sec1)               |
| 2.    | [Chapter I-1: The ML Landscape](#sec2) |

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

However, instead of learning about the problem ourselves, identify patterns to solve ourselves, and specific solutions to each problems ourselves, we can get a computer to do it, faster and more straightforwardly with all the 21st century computational power made available to us. ***This is usually shorter, easier to mantain, and more accurate.***

---
[](Reference!------)  

[2]: https://www.coursera.org/course/neuralnets
[3]: http://shop.oreilly.com/product/0636920033400.do
[book-github]: https://github.com/ageron/handson-ml2 
[i1]: https://i.imgur.com/DhjVk61.png "source - https://imgur.com/a/qvyuhVh"