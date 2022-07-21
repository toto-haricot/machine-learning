# Naïve Bayes Classifier

## Using this package 🔑

*coming soon*

## Matematics and assumptions 👨‍🏫

### Notations ✏️

Let's first introduce some notations to clarify our classification problem. 

- $x$ is any input and its number of features is $d$. So any observation can be expressed as : <br>

$$x = (x_1, x_2, ... , x_d)$$

- $K$ is the number of class and $C_k$ denotes one the $K$ classes. The class associated to any observation is noted y and so:

$$y \in (C_1, C_2, ... , C_K)$$

- $\sigma$ and $\mu$ represents standart deviations and means of any data serie <br>

- $P(C_k|x)$ is the probability of $C_k$ given that $x$ happens. In order words it's the probability that observation $x$ has class $C_k$ <br><br>

### Naïve Bayes Theorem 💡

As its name can tell, a Naïve Bayes classifier relies upon the **Naïve Bayes Theorem**. As our aim is to predict the class of a given $x$, we will use the Naïve Bayes Theorem to express the propability of each class given $x$ : <br>

$$P(C_k|x) = \frac{P(x|C_k)P(C_k)}{P(x)}$$

Then we have three terms to estimate : $P(x|C_k)$, $P(C_k)$ and $P(x)$. We will learn an estimation of each of these terms thanks to training data. <br><br>

### Learning parameters 🍃

<ins>$P(x|C_k)$</ins> : To compute this term we will use the **naïve hypothesis** which assumes that the data features are independents. That means that for any $i$ and $j$, $x_i$ and $x_j$ are assumed as independent. Always keep in mind that this a huge assumption that doesn't suit all types of datasets ! 
Thus we can express $P(x|C_k)$ such as : 

$$P(x|C_k) = \prod_{i=1}^{d}P(x_i|C_k)$$

And to know $P(x_i|C_k)$ we will make another assumption which is to say that the distributions of $x_i$ in each class has a Gaussian Probability Density Function. 

$$P(x_i|C_k) = \frac{1}{\sqrt{2\pi}\sigma}\exp{(-\frac{(x-\mu)^2}{2\sigma^2})}$$

Then we just have to find $\mu$ and $\sigma$ for every data feature and every class.<br><br>

<ins>$P(C_k)$</ins> : It will just be the number of times we observe class $C_k$ in training data divided by the number of training samples. <br><br>

<ins>$P(x)$</ins> : It is just a normalization constant so that we indeed have a probability. Our aim is only to find out $k$ for which $P(C_k|x)$ is maximum so we will drop that term for now. <br><br>

## Requirements ☝️

