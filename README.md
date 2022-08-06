# Machine Learning 👩‍💻

In this repository we present an **implementation from scratch**, using only `numpy` and python most basic libraries, of the classical machine learning algorithms. Here comes the list of all algorithms we wish to implement : 

- [x] [K-Means](#k-means-)
- [x] [Linear Regression](#linear-regression-)
- [ ] [Linear Regression Regularized](#linear-regression-regularized-)
- [x] [Logistic Regression](#logistic-regression-)
- [x] [Linear Discriminant Analysis](#linear-discriminant-analysis-)
- [ ] [Multi Layer Perceptron](#multi-layer-perceptron-)
- [x] [Naïve Bayes](#naïve-bayes-)
- [ ] [Quadratic Discriminant Analysis](#quadratic-discriminant-analysis-)
- [x] [Random Forest](#random-forest-)

For each algo we create a specific repository in which you can find the python implementation from scratch along with a jupyter notebook which is meant to train and test our code on some open source datasets. To provide a better understanding of the algorithm implemented, you will find in each repository a `readme.md` file that goes through the matematics that run the algo. <br><br>

In the `datasets/` folder you will find several classic open source datasets that we will use to train and test our models. In addition a python module called `utils.py` gives some useful functions to work with the datasets.<br><br>

<ins>Note</ins> : In this repository we will propose an implementation for vanilla neural network in the `mlp` (multi layer perceptron) repository but we won't go further into deep learning. Deep learning projects will be sharing in another coming repository. <br>

# Models 🗳️
## K-Means ✅

K-Means is a very simple-to-understand **clustering algorithm**. We start by setting the parameter K which represents the number of clusters we are looking for. Then we initialize K points at random as our clusters **centroïds**. As its name suggests, a centroïd is simply the center point of a cluster. Once our centroïds are randomly choosen, we compute for each point its Euclidien Distances to the centroïds. We form the clusters by assigning each point to its closest centroïd. After that we get K groups of data and we will compute the centers of these clusters which we will assign as the new centroïds. Then we can once again form the clusters, and compute the centroïds, and form new clusters, and compute the new centroïds and so on...<br>

More details and illustrations on the K-Means algorithm will soon be available in the coming `k_mean/readme.md` file. <br><br>

## Linear Regression ✅

Linear regression is probably the most common machine learning algorithm. Most of us had already used it even before starting to learn data science or artificial intelligence. This algorithm deals with <font color="orange"> **regression** </font> problems and it can also be applied to classification but it's less relevent. The assumption made is that the output $y$ of an input $x$ is linear combination that input with a set of parameters $\omega$. <br><br>

Our code is done in a way that allows the model to be used for multi-linear regression. <br><br>

## Linear Regression Regularized 🚧

*Coming soon* <br><br>

## Logistic Regression ✅

Logistic regression is probably the most famous classification algorithm. It is quite similar to linear regression except that we pass its result into a sigmoïd function that resizes the output between 0 and 1. We can then interpret this result as a probability and assign to any $x$ the class $C_k$ with the highest probability. Once again, please refer to the `logistic_regression/readme.md` file to have the global matematical overview. <br>

<ins>Note</ins> : The current implementation only allows binary classification. This is quite a restriction so we will soon improve it to be multi-class compatible. <br><br>

## Linear Discriminant Analysis ✅

*Code available and description coming very soon* <br><br>

## Multi Layer Perceptron 🚧

*Coming soon* <br><br>

## Naïve Bayes ✅

*Code available and description coming very soon* <br><br>

## Quadratic Discriminant Analysis 🚧

*Coming soon* <br><br>

## Random Forest ✅

*First draft of code available and description coming very soon* <br><br>

blabla
