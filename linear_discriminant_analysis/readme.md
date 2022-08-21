# Linear Discriminant Analysis

## Using this package 🔑

*coming soon*

## Algorithm explained 👨‍🏫

Linear Discriminant Analysis, commonly called LDA, is a supervised classification algorithm. <br>

### LDA illustrated 🖼️

Before going through the mathematics that run the LDA algorithm, I propose you to have a look to how LDA operates on
a 2 dimensional example with binary classes. Let's assume that we have a dataset whose training set look like the following : 

<img src="https://github.com/toto-haricot/machine_learning/blob/master/_illustrations/LDA_readme_01.png" width="550">

The idea of Discriminant Analysis (LDA and Quadratic Discriminant Analysis which is presented in another repository) is to 
model each class by a Gaussian distribution. The aim is to draw some regions for each class and then when we will try to 
predict the class of a new data point later on, we will just have the see in which regions it is located. To draw thoses 
regions based with Gaussians shapes, we just have to compute the means and covariances (we'll explain how to do so in the 
next section) of each class. This would give something like this : 

<img src="https://github.com/toto-haricot/machine_learning/blob/master/_illustrations/LDA_readme_02.png" width="550">

If we keep the gaussians under this form we are entering the field of Quadratic Disciminant Analysis. In **Linear** Discriminant
Analysis we make the assumption that each attribute, for all classes, have the same variances. This will force the gaussian
shapes to be similar for all classes and our regions now looks like the following : 

<img src="https://github.com/toto-haricot/machine_learning/blob/master/_illustrations/LDA_readme_03.png" width="550">

The covariance choosen to design the gaussians is the between class scatter matrix which is nothing else than the weighted sum
of the matrix covariances of each class. Once again the exact arithmetics to compute this matrix will be presented in the 
coming section. 

Ok great, but what if I want to classify a new data point ? It's still the very purpose of a classification algorithm right. 
How can thoses gaussians help us to decide to which class I should assign a new observation ? 

<img src="https://github.com/toto-haricot/machine_learning/blob/master/_illustrations/LDA_readme_04.png" width="550">

Well as you can guess, we will assign this new green point to the closest class, ie. to the closest gaussian. To do so we will 
actually draw a decision boundary and depending where the point is located in respect to this boundary we will assign it one 
of the classes. 

<img src="https://github.com/toto-haricot/machine_learning/blob/master/_illustrations/LDA_readme_05.png" width="550">

<br>

### Notations and problem statement ✏️

Now that we have a better idea of how LDA works, let's go through how the algorithm really operates so that we finally have
everything we need to implement it from scratch. 

**Problem statement** : Let's suppose that we have a $K$ class classification problem. Our data set is called $X$ and its
number of data points is $n$. The training and testing sets are $X_{train}$ and $X_{test}$ with respective lengths of 
$n_{train}$ and $n_{test}$. The number of features is $d$ (just like d) and the classes are noted $c_k$ with $k$ between 1
and $K$. The vector containing the classes, sometimes also refered to as labels, is noted $y$ and each $y_i$ is equal to a
certain $c_k$. 

$$X = \begin{bmatrix}
x_{1,1} & x_{1,2} & ... & x_{1,d} \\
x_{2,1} & x_{2,2} & ... & x_{2,d} \\
... & ... & ... & ... \\
x_{n,1} & x_{n,2} & ... & x_{n,d}
\end{bmatrix}$$

$$Y = \begin{bmatrix}
y_{1} \\
y_{2} \\
... \\
y_{n}
\end{bmatrix}$$

<br>

### Finally some maths 🧮

LDA is a discriminative model, that means that it will learn the boundary between classes so that when it will have to predict
the label of a new data point it will be able to give the probability of the point to belong to each class. Then it will just
assign the point to the most likely class. Ok so we want to be able to compute the probability to belong to each class for any
data point $x$, let's try to express $P(Y = c_k | X = x)$ ie. the probability that $x$ belong to the class $k$.

The **Bayes theorem** states that : 

$$P(Y = c_k | X = x) = \frac{P(X = x | Y = c_k).P(Y = c_k)}{P(X = x)}$$

Let's see how we can express each term one by one. 

- $P(X = x | Y = c_k)$ is the likelihood. We have an expression thanks to the fact that we have assumed that the classes 
followed Gaussian distributions. We know that the density function of a gaussian distribution is such as :

$$P(X = x | Y = c_k) = \frac{1}{\sqrt{2\pi\Sigma_k}} \exp{(-\frac{1}{2}(x - \mu_k)^{T}\Sigma_k^{-1}(x - \mu_k))}$$

And we can see that the only things we need to determine are $\mu_k$ and $\Sigma_k$

- ...









