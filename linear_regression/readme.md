# Linear Regression

## Using this package 🔑

*coming soon*

## Matematics and assumptions 👨‍🏫

### Notations ✏️

Let's first introduce some notations to clarify our regression problem. Linear Regression is all about finding a set of parameters, that we note $\omega = (\omega_0, \omega_1, \omega_2, ... , \omega_d)$, which can map an input $x = (x_1, x_2, ... , x_d)$ to a corresponding target $y$ via a simple **linear combination**. The relation between $x$ and $y$ is then given by: <br>

$$y = \omega_0 + x_1.\omega_1 + x_2.\omega2 + ... + x_d.\omega_d$$ <br>

Here we will assume that $y$ is of dimension 1 and $x$ can be one any side, that we note $d$ (just like dimension). <br><br>

The challenge is then to find the set of $d$ parameters $\omega = (\omega_0, \omega_1, \omega_2, ... , \omega_d)$ that are the most accurate at computing $y$ for a given $x$. Let's see how to compute $\omega$ <br><br>

As always in machine learning, we study a dataset and we aim at finding a pattern in that dataset. In our case were a looking for a linear correlation between some inputs $x$ and corresponding outputs $y$. <br>

Let's say that we have a dataset $X$, composed of n observations $x_i$ and each observation is of dimension $d$. We can express $X$ such as : <br>

$$X = \begin{bmatrix}
x_{1,1} & x_{1,2} & ... & x_{1,d} \\
x_{2,1} & x_{2,2} & ... & x_{2,d} \\
... & ... & ... & ... \\
x_{n,1} & x_{n,2} & ... & x_{n,d}
\end{bmatrix}$$

For each of the n observations $x_i$ that compose $X$ we know the corresponding outputs $y$ stored in the vector $Y$ which is such as : <br>

$$Y = \begin{bmatrix}
y_{1} \\
y_{2} \\
... \\
y_{n}
\end{bmatrix}$$

We then split our dataset between a training and testing set. From now on, we only consider the training set which is composed of $n_{train}$ observations among the $n$. A common value for $n_{train}$ is to be 80% of $n$, in other words we train our model on 80% of the available observations and keep the remaining 20% for testing. Now let's see how to compute $\omega$ thanks to the training dataset $X_{train}$. <br><br>

### Learning parameters 🍃

We want $\omega$ to give us the best predictions possible. That means that for each element of $X_{train}$, for instance $x_i$, we want $y_{i,pred} = \omega_0 + x_{i,1}.\omega_1 + x_{i,2}.\omega2 + ... + x_{i,d}.\omega_d$ to be as close as possible as ground truth $y_i$. <br><br>

To measure this notion of "as close as possible", we usually try to minimize the square distance between the two scalars $y$ and $y_{pred}$ : 

$$d(y, y_{pred}) = (y - y_{pred})^2$$ <br>

We want this distance to be minimal for all predictions of the training dataset. So we are now able to state the optimization problem we want to solve : 

$$\min_{\omega_0, \omega_1, \omega_2, ... , \omega_d} \sum_{i=1}^{n_{train}}(y_i - y_{pred,i})^2$$ <br>

To solve this problem, ie. to find the optimal set of parameters $\omega = (\omega_1, \omega_2, ... , \omega_d)$ we can define the loss function $J(\omega)$ such as : 

$$J(\omega) = \sum_{i=1}^{n_{train}}(y_i - y_{pred,i})^2$$

Let's jump to the matrix notation for more efficient computations. 

$$Y_{pred} = \begin{bmatrix}
y_{pred, 1} \\
y_{pred, 2} \\
... \\
y_{pred, n}
\end{bmatrix} = \begin{bmatrix}
\omega_0 + x_{1,1}.\omega_1 + x_{1,2}.\omega2 + ... + x_{1,d}.\omega_d \\
\omega_0 + x_{2,1}.\omega_1 + x_{2,2}.\omega2 + ... + x_{2,d}.\omega_d \\
... \\
\omega_0 + x_{n,1}.\omega_1 + x_{n,2}.\omega2 + ... + x_{n,d}.\omega_d
\end{bmatrix}$$

That we can split into matrices that contain either terms of $X_{train}$ or terms of $\omega$ : 

$$Y_{pred} = \begin{bmatrix}
1 & x_{1,1} & x_{1,2} & ... & x_{1,d} \\
1 & x_{2,1} & x_{2,2} & ... & x_{2,d} \\
... & ... & ... & ... & ... \\
1 & x_{n,1} & x_{n,2} & ... & x_{n,d}
\end{bmatrix} . \begin{bmatrix}
\omega_0 \\
\omega_1 \\
... \\
\omega_d
\end{bmatrix} = \hat{X} . \Omega$$

<ins>Note</ins> : We can see that the first matrix is not exactly $X$ (or $X_{train}$ if we consider the training set) but $X$ concatenated with a column of ones, this is why we use the notation $\hat{X}$. This enables to take into account $\omega_0$ in the predictions. $\omega_0$ is called the **biais** and the remaining $\omega_i$ are called the **weights**. <br><br>

Eventually we can rewrite the previously defined loss function with matrices : <br>

$$J(\Omega) = (Y - Y_{pred})^T.(Y - Y_{pred}) = \|\vert Y - \hat{X}\Omega\|\vert^2$$ <br>

Now we can compute the gradient of $J(\Omega)$ with respect to $\Omega$


