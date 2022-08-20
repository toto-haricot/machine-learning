# Linear Discriminant Analysis

## Using this package 🔑

*coming soon*



## Algorithm explained 👨‍🏫

Linear Discriminant Analysis, commonly called LDA, is a supervised classification algorithm.

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

### Notations and assumptions ✏️

LDA make the two following assumptions : 

- Each class follows a Gaussian distribution
- All attributes have the same variance

The aim of the training of LDA is then to approximate the parameters of each class. Assuming that these classes are 
Gaussian shaped, the parameters to determine are the mean $\mu_{k}$ with $k$ one of the $K$ classes and the covariance 
$\sigma$. The fact that $\sigma$ does not depend on $k$ is due to assumption (2). Indeed in LDA we suppose that the 
variance of all gaussian (that represent the classes) is the same. That means that the shapes of gaussians will all be
the same, only the position (parametrized by $mu_k$) will change. We can illustrate that with the following graph. 



Let's assume that we have already divided our dataset into training and testing sets and so we can use the training 
set $X_{train}$ to estimate the aforementionned parameters. The estimation of $\mu_{k}$ is pretty straightforward : 

$$ \mu_{k} = 1/n_k \sum_{x_i \in k} x_i$$

With $n_k$ the number of data points in the class $k$. And the estimation of the covariance $\sigma_k$ is given by : 

$$  $$



