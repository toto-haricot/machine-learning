# Machine Learning 👩‍💻

The aim of this repository is to **implement from scratch**, using only `numpy` and python most basic libraries, the classical machine learning algorithms. Here comes the list of all algorithms we wish to implement : 
- [x] [Linear Regression](#linear-regression-)
- [x] [Linear Regression Regularized](#linear-regression-regularized-)
- [x] [Logistic Regression](#logistic-regression-)
- [x] [Linear Discriminant Analysis](#linear-discriminant-analysis-)
- [ ] [Multi Layer Perceptron](#multi-layer-perceptron-)
- [x] [Naïve Bayes](#naïve-bayes-)
- [ ] [Quadratic Discriminant Analysis](#quadratic-discriminant-analysis-)
- [x] [Random Forest](#random-forest)

For each algo we create a specific repository in which you can find the python implementation from scratch along with a jupyter notebook which is meant to train and test our code on some open source datasets. To provide a better understanding of the algorithm implemented, we provide in each repository a `readme.md` file that states the matematics that run the algo. <br><br>

In the `datasets/` folder you will find several classic datasets that are used to train and test some machine learning models. <br><br>

<ins>Note</ins> : In this repository we will propose an implementation for vanilla neural network in the `mlp` (multi layer perceptron) repository but we won't go further into deep learning. Deep learning projects will be sharing in another coming repository. <br>

# Models 🗳️
## Linear Regression ✅

Linear regression is probably the most common machine learning algorithm. Most of us had already used it even before starting to learn data science or artificial intelligence. This algorithm deals with <font color="orange"> **regression** </font> problems and it can also be applied to classification but it's less relevent. The assumption made is that the output $y$ of an input $x$ is linear combination that input with a set of parameters $\omega$. <br><br>

Our code is done in a way that allows the model to be used for multi-linear regression. <br>

## Linear Regression Regularized 🚧

*Coming soon* <br><br>

## Logistic Regression ✅

Logistic regression is probably the most famous classification algorithm. It is quite similar to linear regression except that we pass its result into a sigmoïd function that resizes the output between 0 and 1. We can then interpret this result as a probability and assign to any $x$ the class $C_k$ with the highest probability. Once again, please refer to the `logistic_regression/readme.md` file to have the global matematical overview. <br>

<ins>Note</ins> : The current implementation only allows binary classification. This is quite a restriction so we will soon improve it to be multi-class compatible. <br><br>

## Linear Discrimimant Analysis ✅

*Code available and description coming very soon* <br><br>

## Multi Layer Perceptron 🚧

*Coming soon* <br><br>

## Naïve Bayes ✅

*Code available and description coming very soon* <br><br>

## Quadratic Dicriminant Analysis 🚧

*Coming soon* <br><br>

## Random Forest ✅

*First draft of code available and description coming very soon* <br><br>
