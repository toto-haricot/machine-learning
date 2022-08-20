# Linear Discriminant Analysis

## Using this package 🔑

*coming soon*

## Algorithm explained 👨‍🏫

Linear Discriminant Analysis, commonly called LDA, is a supervised classification algorithm.

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



