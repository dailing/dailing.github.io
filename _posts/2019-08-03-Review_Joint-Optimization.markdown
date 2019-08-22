---
layout: post
title:  "Learning With Noisy Labels"
date:   2019-08-03 12:10:09 +0800
categories: deeplearning
---

DNN trained on large-scale dataset achieves have achieved impressive result on many problems (classification, detection and segmentation).
In almost all projects, data matters!
Accurate labels are so important for DNN training.
While, noise in labels is inevitable, especially in medical dataset.
How to model the noise and distinguish the right from wrong is important during the training.

**What we've got:**
* the dataset is dirty (label noisy)
* noisy distribution is unknown.
* we can have someone rechecking the label, but expensive

**What we want:**
* A model that predicts well
* The model should give clue about the prediction confidence
* The model generalize well even trained with noisy label 
* The model can identify obvious errors and give possible errors 


# Generalization abilities of DNN
> Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (ICLR 2016). Understanding deep learning requires rethinking generalization. 1–15. Retrieved from http://arxiv.org/abs/1611.03530

Networks are complex enough to memorize all labels of large-scale training data.
The author tried to train networks with random training label, 
the networks converged well finally by memoriezing all labels. 
In such extream conditions,
the generalization loss is large and the result makes none-sence.
From this work we can conclude that networks converges with noisy labels.
Adding any form of regularization (data augmentation, weigh decay, drop) would not affect the final convergence.
The problem is: how to effectively model the noise and avoid the network from memorizing labels.

Some interesting results from Zhang's work:
- Deep neural networks easily fit random labels.
- Explicit regularization may improve generalization performance, but is neither necessary nor by itselfsufficient for controlling generalization error.
- Data augmentation is the most effective regularization among all regularization discussed in this paper.


> Arpit, D., Jastrzębski, S., Ballas, N., Krueger, D., Bengio, E., Kanwal, M. S., … Lacoste-Julien, S. (2017). A Closer Look at Memorization in Deep Networks. Retrieved from http://arxiv.org/abs/1706.05394

This paper also claim the DNNs trained with SGD-variants first use patterns, then brute force memorization, to fit the data.

So, theoretically, training DNN on noisy dataset to obtain correct classification is possible. The problem is: how to make DNNs learn patterns and ignore the noise labels, or how to enforce the pattern learning and weaken the memorizing.

Two ways I can think of to handle this noisy label problem:
* explicitly model the noise and add it to the model
  * loss function is the first place one can think of
  * other ideas such as soft label?
* modify the dataset using the model to get ride of the noise
  * deleting samples 
  * separating the dataset
  * casecade network
  * modifying the labels

# Noise transition matrix
Let $l$ and $l^{GT}$ be the noisy and true labels. The noise transition matrix $T$ can be defined as:

$$ t_{ij} = p(l = j|l^{GT}=i) $$

And the cross entropy loss

$$ \mathcal{L}(\boldsymbol{\theta}, Y, X)=-\frac{1}{n}\sum_{i=1}^n \boldsymbol{y_i}^{\mathrm{T}} \log(s(\boldsymbol{\theta}, \boldsymbol{x}_i)) $$

 can be modified to:

$$
\mathcal{L}(\boldsymbol{\theta}, Y, X)=-\frac{1}{n} \sum_{i=1}^{n} \log \left(\boldsymbol{y}_{i}^{\mathrm{T}}Ts\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)
$$

Maybe this form works too:

$$
\mathcal{L}(\boldsymbol{\theta}, Y, X)=-\frac{1}{n} \sum_{i=1}^{n} \boldsymbol{y}_{i}^{\mathrm{T}}T \log \left(s\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)
$$


This modified cross entropy is useful when the label are noisy. 
When one-hot coding is used, $$\boldsymbol{y}_{i}$$ select a row from $T$ and the selected row $$T_{y_i}$$ then dot product with prediction $s(\boldsymbol{\theta}, \boldsymbol{x}_{i})$. 
It can be seen as embeding of label $y_i$, since $y_i$ actually selects a row from $T$. 
It is interesting because we can use $T$ to model the noise distribution. 
By careful selecting $T$, the erroneous labels will get less attention.

Some work presuppose the ground truth noise transition matrix $T$. While the distribution of label noise  is unknown, I don't think this way works. Other method model $T$ by a fully connected layer and trained end to end.

More on this!

> S. Sukhbaatar, J. Bruna, M. Paluri, L. Bourdev, and R. Fergus. Training convolutional networks with noisy labels. In ICLR, 2015

> I. Jindal, M. Nokleby, and X. Chen. Learning deep networks from noisy labels with dropout regulariza- tion. In ICDM, 2016


# noise-tolerance loss function
Loss functions such as ramp loss, unhinged loss, MSE and MAE can be used to train more robust models. But it should be noted that DNN can learn random labels by memorizing. The power of loss is limited.

## Weighted Approximate-Rank Pairwise (WARP) Loss
> Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling up to large vocabulary image annotation. IJCAI International Joint Conference on Artificial Intelligence, 2764–2770. https://doi.org/10.5591/978-1-57735-516-8/IJCAI11-460

Consider the task of ranking labels $i \in \mathcal{Y}$ given an example x. In our setting labeled pairs $(x, y)$ will be provided for training where only a single annotation $y_i \in \mathcal{Y}$ is labeled correct. Let $f(x) \in \mathbb{R}^{Y}$ be a vector function providing a score for each of the labels, where $f_i(x)$ is the value for label $i$. Ranking error functions is defined as:

$$ err(f(x),y) = L(\operatorname{rank}_y(f(x))) $$

$$
\operatorname{rank}_{y}(f(x))=\sum_{i \neq y} I\left(f_{i}(x) \geq f_{y}(x)\right)
$$

$$L(k)=\sum_{j=1}^k\alpha_j, \text{with } \alpha_1 \geq \alpha_2 \geq \cdots\geq 0$$

- $\operatorname{rank}_y(f(c))$ is the rank of the true label $y$ given by $f(x)$.
- $I$ is the indicator function.
- $L(\cdot)$ transform the rank into a loss
- $\alpha_j=\frac{1}{j}$ yield the best result.

The loss function is equal to 

$$err(f(x),y) = \sum_{i \neq y}L(\operatorname{rank}_y(f(x)))\frac{I(f_i(x) \geq f_y(x))}{\operatorname{rank}_y(f(x))}$$

- $0/0=0$ when the correct  label $y$ is top ranked.

Note that this formulation is not differentiable.
Using the hinge loss instead of the indicator function to add a margin and make the loss continuous, $err$ can be approximated by:

$$
\overline{\operatorname{err}}(f(x), y)=\sum_{i \neq y} L\left(\operatorname{rank}_{y}^{1}(f(x))\right) \frac{\left|1-f_{y}(x)+f_{i}(x)\right|_{+}}{\operatorname{rank}_{y}^{1}(f(x))}
$$

$$
\operatorname{rank}_{y}^{1}(f(x))=\sum_{i \neq y} I\left(1+f_{i}(x)>f_{y}(x)\right)
$$

$$
\operatorname{Risk}(f)=\int \overline{\operatorname{err}}(f(x), y) d P(x, y)
$$

- $|t|_+$ is the positive part of $t$
- $\operatorname{rank}_{y}^{1}(f(x))$ is the margin penalized rank of $y$
- $\operatorname{Risk}(f)$ is what we want to minimize

An unbiased estimator of this risk can be obtained by stochastically sampling. To do this:
1. Sample a pair $(x, y)$ according to $P(x, y)$
1. For the chosen $(x, y)$ sample a violating label $\overline{y}$ such that $1 + f_{\overline{y}}(x) > f_y(x)$


![](/asserts/post/2019-08-03-learning-with-noisy-labels/wrap_loss_1.png)


more on this!

# Joint Optimization Framework for Learning with Noisy Labels
> Tanaka, D., Ikami, D., Yamasaki, T., & Aizawa, K. (2018). Joint Optimization Framework for Learning with Noisy Labels. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 5552–5560. https://doi.org/10.1109/CVPR.2018.00582

## Framework
![Frame work](/asserts/post/2019-08-03-learning-with-noisy-labels/joint_optim_1.png)

The overall training procedure can be divided into tow step, initial training and training with modified labels:

1. This method first train network with large learning rate (so that DNN learn patterns rather than memorizing noisy labels).
   - Labels are updated (starting from the 70th epoch) using the result of trained network.
   - label $\boldsymbol{y}_i$ are updated with probability $\boldsymbol{s}$
   - average output probability of the past 10 epoch are used as $\boldsymbol{s}$
2. Train the network again using the new labels
   - learning rate is decreased in this stage to better fit to data.
   - only $\mathcal{L}_c$, classification loss, is used in this stage.

## Optimization 

### loss function 
The loss function contains three parts, classification loss and tow regularization loss.

$$
\mathcal{L}(\boldsymbol{\theta}, Y | X)=\mathcal{L}_{c}(\boldsymbol{\theta}, Y | X)+\alpha \mathcal{L}_{p}(\boldsymbol{\theta} | X)+\beta \mathcal{L}_{e}(\boldsymbol{\theta} | X)
$$

- $\mathcal{L}_{c}$ denote the classification loss.
- $\mathcal{L}_{p}$ prevent a trival minimum solution by assigning all labels to a single class.
- $\mathcal{L}_{e}$ prevent local minimum when soft labels are used.

### classification loss
In this study, the Kullback-Leibler (KL)-divergence is used for classification loss.

$$
\mathcal{L}_{c}(\boldsymbol{\theta}, Y | X)=\frac{1}{n} \sum_{i=1}^{n} D_{K L}\left(\boldsymbol{y}_{i} \| \boldsymbol{s}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)
$$

$$
D_{K L}\left(\boldsymbol{y}_{i} \| \boldsymbol{s}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)=\sum_{j=1}^{c} y_{i j} \log \left(\frac{y_{i j}}{s_{j}}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)
$$

There are some concerns about the formulation.

What if $y_{ij}=0$? The author didn't explicitly stat how to handle $D_{KL}$ when $y_{ij}=0$. Since $\lim_{x \to 0^+}x\log(x) = 0$, here I assume that $y_{i j} \log \left(\frac{y_{i j}}{s_{j}}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)\right)=0$ if $y_{ij}=0$.


> https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

In mathematical statistics, the Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is different from a second, reference probability distribution.

For discrete probability distributions ${\displaystyle P}$ and ${\displaystyle Q}$ defined on the same probability space, the Kullback–Leibler divergence between ${\displaystyle P}$ and ${\displaystyle Q}$ is defined to be

$$
D_{\mathrm{KL}}(P \| Q)=-\sum_{x \in \mathcal{X}} P(x) \log \left(\frac{Q(x)}{P(x)}\right)
$$

which is equivalent to 

$$
D_{\mathrm{KL}}(P \| Q)=\sum_{x \in \mathcal{X}} P(x) \log \left(\frac{P(x)}{Q(x)}\right)
$$

In the context of machine learning, ${\displaystyle D_{\text{KL}}(P\parallel Q)}$ is often called the information gain achieved if ${\displaystyle Q}$ is used instead of ${\displaystyle P}$.
In other words, it is the amount of information lost when ${\displaystyle Q}$ is used to approximate ${\displaystyle P}$.
In applications, ${\displaystyle P}$ typically represents the "true" distribution of data, observations, or a precisely calculated theoretical distribution, while ${\displaystyle Q}$ typically represents a theory, model, description, or approximation of ${\displaystyle P}$. In order to find a distribution ${\displaystyle Q}$ that is closest to ${\displaystyle P}$, we can minimize KL divergence and compute an information projection.

One problem is that the KLH  divergence is not symmetric. Kullback and Leibler themselves actually defined the divergence as:

$${\displaystyle D_{\text{KL}}(P\parallel Q)+D_{\text{KL}}(Q\parallel P)}$$

Why Kullback-Leibler (KL)-divergence? The author didn't stat explicitly their reasons for the choise of KL-divergence. From the definition of KL-divergence, we can infer that rather than make binary classifications, the model tries to approximate the distribution of the noisy labels. While the distributions of labels are unknown since there are only one label for each image(one observation per sample), the $y_{ij}$ are then updated to the soft label, which is more like a distribution. In this whole updating procedure, the network predicts the distribution of labels. In the noisy label scenario, predicting the distribution is the best that we can do.

### Regularization loss $\mathcal{L}_p$
A trivial solution of this framework is to assign same label to all training data. $\mathcal{L}_p$ prevent this trivia solution.


\begin{align} 
L_{p} & = D_{\text{KL}}(\boldsymbol{p}\parallel \overline{\boldsymbol{s}}(\boldsymbol{\theta},X))
\\\\  & = \sum_{j=1}^{c} p_{j} \log \frac{p_{j}}{\overline{s}_{j}(\boldsymbol{\theta}, X)}
\end{align} 

$$
\overline{s}(\boldsymbol{\theta}, X)=\frac{1}{n} \sum_{i=1}^{n} s\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right) \approx \frac{1}{|\mathcal{B}|} \sum_{\boldsymbol{x} \in \mathcal{B}} s(\boldsymbol{\theta}, \boldsymbol{x})
$$

- $\boldsymbol{p}$ is the prior probability distribution, which is a distribution of classes among all training data.
- $\overline{s}(\boldsymbol{\theta}, X)$ is the mean probability in trianing data, approximated by performing a calculation for each mini-batch.

So, $\mathcal{L}_p$ is another KL divergence from the predicted distribution to the actual distribution of training label. This regularization make distribution of prediction similar to the actual distribution. Which could be used to avoid the trivial solution.

But the approximation in each batch make it hard to work on dataset with a large number of class or with unbalanced labels. **Oversampling may work in unbalanced situation.**

### Regularization loss $\mathcal{L}_e$
Using soft label cause a problem. $\mathcal{L}_c=0$ become if soft label is used. And then the update stops for these labels.

$$\mathcal{L}_{e}=-\frac{1}{n} \sum_{i=1}^{n} \sum_{j=1}^{c} s_{j}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right) \log s_{j}\left(\boldsymbol{\theta}, \boldsymbol{x}_{i}\right)$$

The plot of $-x\log(x)$ is like this:
![Frame work](/asserts\post\2019-08-03-learning-with-noisy-labels\joint_optim_2.png)

So basically this regularization will push $s_{j}(\boldsymbol{\theta}, \boldsymbol{x}_{i})$ towards 0 or 1. But in fact, both $\mathcal{L}_p$ and $\mathcal{L}_e$ will make the optimization out of local minimum. And there is no experiment to prove that $\mathcal{L}_e$ actually help with the situation.

# Nonlinear, Noise-aware, Quasi-clustering Approach to Learning Deep CNNs from Noisy Labels

> Cvpr, A., & Id, P. (2019). A Nonlinear , Noise-aware , Quasi-clustering Approach to Learning Deep CNNs from Noisy Labels. CVPR.

The author presented a Nonlinear, Noise-aware, Quasi-clustering (NNAQC), a method for learning deep convolutional networks from datasets corrupted by unknown label noise.
## Framework
![Frame work](/asserts\post\2019-08-03-learning-with-noisy-labels\NNAQC.png)

$$
\mathcal{L}\left(\Theta, W ; \mathcal{D}^{\prime}\right)=\alpha \mathcal{L}_{\mathrm{NNA}}\left(\Theta, W ; \mathcal{D}^{\prime}\right)+(1-\alpha) \mathcal{L}_{\mathrm{QC}}\left(\Theta ; \mathcal{D}^{\prime}\right)
$$

$$
\begin{aligned} \mathcal{L}_{\mathrm{NNA}}\left(\Theta, W ; \mathcal{D}^{\prime}\right) &=-\frac{1}{n} \sum_{i=1}^{n} \log p\left(y^{\prime}=\hat{y}_{i}^{\prime} | x_{i} ; \Theta, W\right) \\ &=-\frac{1}{n} \sum_{i=1}^{n} \log \left[\sigma\left(W \sigma\left(t_{1}\left(x_{i} ; \Theta\right)\right)\right)\right]_{\hat{y}_{i}} \end{aligned}
$$

$$
\begin{aligned} \mathcal{L}_{\mathrm{QC}}\left(\Theta ; \mathcal{D}^{\prime}\right)=-& \frac{1}{n} \sum_{i=1}^{n}\left(\beta p\left(\hat{y}_{i} | x_{i} ; \Theta\right)+(1-\beta) y_{i}^{\prime}\right)  \times \log p\left(\hat{y}_{i} | x_{i} ; \Theta\right) \\=-& \frac{1}{n} \sum_{i=1}^{n}\left(\beta\left[\sigma\left(t_{1}(x ; \Theta)\right)\right)\right]_{\hat{y}_{i}}+(1-\beta) y_{i}^{\prime} \times \log \left[\sigma\left(t_{1}(x ; \Theta)\right)\right]_{\hat{y}_{i}} \end{aligned}
$$

* $f_1(\Theta, x)$: CNN
* $\sigma(.)$: softmax
* $y$: true label
* $y'$: noisy label
* $\hat{y}$: predicted true label probability
* $l_1(\hat{y}, W) = W\times \hat{y}$: probability transfer function
* $\hat{y}'$: predicted noisy label probability
* $\mathcal{L}_{\mathrm{NNA}}$: Nonlinear, Noise-Aware loss
* $\mathcal{L}_{\mathrm{QC}}$: Quasi-Clustering loss

The $l_1$ part can be implemented by fully connected layer without bias. The overall $\mathcal{L}$ is differentiable.

## None-linear
![None-linear](/asserts\post\2019-08-03-learning-with-noisy-labels\none_linear.png)
![Linear](/asserts\post\2019-08-03-learning-with-noisy-labels\linear.png)

The author derived the gradient of loss $\mathcal{L}$ with respect to $\Theta$ to show the in-depth view of this method.

For none-linear transfer proposed in this paper(upper picture):

$$
\begin{aligned} \frac{\partial \mathcal{L}}{\partial \Theta} &=\frac{\partial \mathcal{L}}{\partial \hat{y}^{\prime}} \frac{\partial \sigma\left(t_{2}\right)}{\partial t_{2}} \frac{\partial l_{1}(\hat{y}, W)}{\partial \hat{y}} \frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}} \frac{\partial f_{1}(\Theta, x)}{\partial \Theta} \\ &=\frac{\partial \mathcal{L}}{\partial \hat{y}^{\prime}}\left(\frac{\partial \sigma\left(t_{2}\right)}{\partial t_{2}} W \frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}}\right) \frac{\partial f_{1}(\Theta, x)}{\partial \Theta} \end{aligned}
$$

For linear transfer normally used(lower picture):

$$
\begin{aligned} \frac{\partial \mathcal{L}}{\partial \Theta} &=\frac{\partial \mathcal{L}}{\partial \hat{y}^{\prime}} \frac{\partial l_{1}(\hat{y}, \sigma(W))}{\partial \hat{y}} \frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}} \frac{\partial f_{1}(\Theta, x)}{\partial \Theta} \\ &=\frac{\partial \mathcal{L}}{\partial \hat{y}^{\prime}}\left(\sigma(W) \frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}}\right) \frac{\partial f_{1}(\Theta, x)}{\partial \Theta} \end{aligned}
$$

According to the author:
> We find that the NNAQC denoising operator is more diffuse than the linear noise model. A more diffuse operator allows for more flexibility in
handling disagreements between the CNN model predic- tions and the noisy labels. 

Notice the middle part:

$$\frac{\partial \sigma\left(t_{2}\right)}{\partial t_{2}} W \frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}}$$

Combination of $\frac{\partial \sigma\left(t_{2}\right)}{\partial t_{2}}$ and $\frac{\partial \sigma\left(t_{1}\right)}{\partial t_{1}}$ wipes out most of the gradient when $\hat{y}$ and $y'$ disagree.

Also the NNAQC prevent overconfident when prediction and the label agrees.

# Some of my thoughts

## On calculation of variation
Recently, image prior work (Ulyanov D., Vedaldi A., L. V. (2018). Deep Image Prior. Cvpr) shows the the CNN architecture seems to learn the images first and then overfitting occurs alone the optimizing steps. I'm not sure if this occurs in classification too. Assume that the network learn useful things first and then overfit noise, the errors and predictions alone the training step seems to be a good source for learning the noisy labels. For example, we can collect the predictions of the training samples, and then use both the prediction and the label as the feature to training a noise model. Some standard and normalized measurement must be designed for this purpose, since it has to be unsupervised.

Bootstrap to get the uncertainty of each samples, and then learns the noisy model?

The noisy model has to be sample-dependent. It best to be sample and class dependent.

## instance dependent transfer function
The easiest way I can think of is to make the transfer matrix a function. 

$$ W = w(\Theta) $$
