
# STAT 535 - Project Report 

#### Author: Dehai Liu    dehail@uw.edu



## Data Description

`Fashion-MNIST` is a dataset of [Zalando](https://jobs.zalando.com/tech/)'s article images—here consisting of a training set of 40,000 examples and a test set of 5,000 examples. Each example is a 28x28 gray scale image, associated with a label from 10 classes. It shares the same image size and structure of training and testing splits.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/data_demo.png" style="zoom:20%" />

<center><B>Fig. 1 </B>First 100 samples of the training set</center>

Each training and test example is assigned to one of the following labels:

| T-shirt | Trouser | Pullover | Dress |    Coat    |
| :-----: | :-----: | :------: | :---: | :--------: |
|    0    |    1    |    2     |   3   |     4      |
| Sandal  |  Shirt  | Sneaker  |  Bag  | Ankle boot |
|    5    |    6    |    7     |   8   |     9      |



## Framework Description

* Scikit Learn 0.20.0 : For the preprocessing and training of decision tree

* MXNet-CU90 1.2.0 Gluon: For the preprocessing and training of LeNet



## Preprocessing

### a. Preprocessing for Decision Tree

#### (1)  Normalization

Supposed the data matrix is $X$, we can normalize it to speed up training by dividing by $255$. 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;X_{norm}=\frac{X}{255}" title="\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" />

#### (2) Max Absolute Scale

To scale each feature to the $[-1,1]$ range without breaking the sparsity of the images, we can use max absolute scaling,
$$
X_{MAbs}=\frac{X_{norm}}{max(abs(X_{norm}),axis=0)}
$$

#### (3) PCA

Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space. Since there are $10$ classes in the dataset, we can first set the decomposition components to be $10$. After careful trial, $10$ tend to be a good choice.



### b. Preprocessing for LeNet

#### (1) Reshape 

Since convolution network requires the input to be $3D$ images, we need to reshape the image vector into image matrix with size of $28\times28$ and one channel. Now the range of each component is $[0,255]$ .

Dimension of data  before reshape: $(40000,784)$

Dimension of data  after reshape: $(40000,28,28,1)$

#### (2) Scaling

Using the function "transform.ToTensor" in Gluon, we can again reshape the data for the standard input in the framework and scale the data in the range $[0,1)$ .

Dimension of data  before scaling: $(40000,28,28,1)$

Dimension of data  after scaling: $(40000,1,28,28)$



## Predictors Description

### a. Decision Tree

#### (1) Model Description$^{[1]}$

Given training vectors $x_i \in R^n$  , $i=1,\cdots, N$ and a label $y $  , a decision tree recursively partitions the space such that the samples with the same labels are grouped together.

Let the data at node $m$ be represented by $Q$. For each candidate split $\theta=(j,t_m)$   consisting of a feature $j$  and threshold $t_m$ , partition the data into $Q_{left}(\theta)$ and $Q_{right}(\theta)$  subsets
$$
\begin{split}
Q_{left}(\theta)&=(x,y)|x_j\leq t_m\\
Q_{right}(\theta)&=Q \backslash Q_{left}(\theta)
\end{split}
$$
The impurity at $m$ is computed using an impurity function $H$  , the choice of which depends on the task being solved (classification or regression)
$$
G(Q,\theta)=\frac{n_{left}}{N_m}H(Q_{left}(\theta))+\frac{n_{right}}{N_m}H(Q_{right}(\theta))
$$
Select the parameters that minimises the impurity
$$
θ^∗=argmin_θ⁡G(Q,θ)
$$
Recurse for subsets $Q_{left}(\theta^∗)$ and $Q_{right}(\theta^∗)$ until the maximum allowable depth is reached, $N_m<min_{samples}$ or $N_m=1$ .



#### (2) Parametrization

|    Parameters    |                       Description                       | Value  |
| :--------------: | :-----------------------------------------------------: | :----: |
|    max depth     |                maximum depth of the tree                | 10:100 |
| min samples leaf | minimum number of samples required to be at a leaf node |  1:5   |



### b. LeNet $ ^{[3]}$

#### (1) Model Description

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/network_2.png" style="zoom:90%" />

<center><B>Fig. 2 </B>Model Architechture (by TensorSpace)</center>

The model is constructed as follows:

|    layer     | filters/channels | kernel size | pool size | strides | activation |
| :----------: | :--------------: | :---------: | :-------: | :-----: | :--------: |
|    input     |        1         |    None     |   None    |  None   |    None    |
|     Conv     |        20        |      5      |   None    |  None   |    Relu    |
|   MaxPool    |        20        |    None     |     2     |    2    |    None    |
|     Conv     |        50        |      5      |   None    |  None   |    Relu    |
|   MaxPool    |        50        |    None     |     2     |    2    |    None    |
| FullyConnect |       120        |    None     |   None    |  None   |    None    |
| FullyConnect |        84        |    None     |   None    |  None   |    None    |
|    output    |        10        |    None     |   None    |  None   |  Softmax   |

#### (2) Parametrization

|  Parameters   |                 Description                  | Value |
| :-----------: | :------------------------------------------: | :---: |
| learning rate |        control the speed for learning        | 0.001 |
|  num epochs   |               number of epochs               |  80   |
|  batch size   | number of  samples in one batch for training |  128  |
|    dropout    |      regularization term in the network      |  0.4  |



## Basic Training Algorithms

### a. Decision Tree

Definition of **Gini**:
$$
Gini(X)=\sum_kp_k(1-p_k)=1-\sum_kp_k^2
$$
Suppose there are $K$ classes in training set $D$, $C_k$ denotes the sample set containing the class $k$. 

The **Gini** of a dataset is:
$$
Gini(D)=\sum_k\frac{|C_k|}{|D|}(1-\frac{|C_k}{|D|})
$$

####  Training Algorithm: CART $ ^{[2]} $

* Input: Training set     $ D={(x_1,y_1),\cdots,(x_N,y_N)} $

  (i) Split the dataset based on the feature $A$, calculate the **Gini** when $A=a$
$$
Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)
$$
  (ii) Pick the feature $A^*$and its corresponding value $a^*$ that minimize the $Gini$ to split the dataset into $D_1$ and $D_2$

  (iii) Loop (i)~(ii) until the stopping condition is satisfied

* Output: Classfication tree $T$



### b. LeNet

#### Training Algorithm: Adam

Adaptive Moment Estimation is a method that computes adaptive learning rates for each parameter. 

Initialize  the following parameter

 $ V_{dw } =0 , V_{d b}=0 , S_{dw}=0, S_{db}=0 $

On iteration $t$ :

  (i) Compute the gradients $dw$ and $db$ using the mini-batch gradient descent

  (ii) $V_{dw}=\beta_1v_{dw}+(1-\beta_1)dw$ , $V_{db}=\beta_1v_{db}+(1-\beta_1)db$

​       $S_{dw}=\beta_1S_{dw}+(1-\beta_1)(dw)^2$,   $ S_{db}=\beta_2S_{db}+(1-\beta_2)(db)^2 $

​       $V_{dw}^{corrected}=\frac{V_{dw}}{1-\beta_1^t}$, $V_{db}^{corrected}=\frac{V_{db}}{1-\beta_1^t} $

​       $S_{dw}^{corrected}=\frac{S_{dw}}{1-\beta_2^t}$, $S_{db}^{corrected}=\frac{S_{db}}{1-\beta_2^t} $

  (iii) Update the weights by
$$
\begin{split}
w:=w-\alpha\frac{V_{dw}^{corrected}}{\sqrt{S_{dw}^{corrected}+\epsilon}}\\
b:=b-\alpha\frac{V_{db}^{corrected}}{\sqrt{S_{db}^{corrected}+\epsilon}}\\
\end{split}
$$
where $\epsilon$ is a small number.



## Training Strategy

### a. Decision Tree

#### (1) Grid Search Cross Validation

* Parameters: minimum samples of leaf and maximum depth
* Training set size: 32000
* Validation set size: 8000
* **5 fold Cross Validation**: record the accuracy of training set and validation set
* Select the model with highest mean accuracy in the validation set

#### (2) Dataset Splitting and model demo

To demonstrate the model, we can again split the data into training set and validation set. At this time, the size of validation set can be much more smaller since it's used for demo instead of tuning the parameter.

*  Training set size: 39600
* Validation set size: 400



### b. LeNet

#### (1) Xavier Initialization

According to the paper of Glorot & Bengio , we assume that for a specific layer $L$, the number of input  and output units are respectively, $n_{in}$ and $n_{out}$ .  

The requirement for the weight $W^L$in layer $L$ should be:
$$
Var(W^L)=\frac{2}{n_{L}+n_{L+1}}
$$
And the initialization of weights $W^L$ follow the uniform distribution:
$$
W\sim U[-\sqrt{\frac{6}{n_L+n_{L+1}}},\sqrt{\frac{6}{n_L+n_{L+1}}}]
$$

#### (2)  Dataset Splitting

Split the data into training set and validation set and shuffle the training set for training.

-  Training set size: 38000
- Validation set size: 2000

#### (3)  Network Architecture and dropout

Following the model description above,  we set up 

* 2 convolution layers with **Relu** activation followed by 2 max pooling layers respectively.
* 2 fully connected layers in the end,  add a **dropout** term serving as **regularization** 

#### (4) Training

Setting the training epochs to be $80$, we record the training loss, validation loss, training accuracy and validation accuracy. Pick the model in the last epoch and save its parameters.



## Experimental Results

### a. Decision Tree

#### (1) Tuning the parameters

The range of the hyper parameter is:

* max_depth: $10,20,30,40,50,60,70,80,90,100$
* min_samples_leaf: $1,2,3,4,5$



For a fixed min samples leaf, we can plot the accuracy vs max depth. Here we set min samples leaf=1.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/gridsearch_depth.png" style="zoom:40%" />

<center><B>Fig. 3 </B>GridSearchCV for max depth in Decision Tree</center>

* The accuracy of training set and validation set both goes up as maximum depth increases and become stable after $max\_depth=40$ .



For a fixed max depth, we can plot the accuracy vs min sample leaf. Here we set max depth=None, which means that the tree will grow to the maximum depth.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/gridsearch_leaf.png" style="zoom:40%" />

<center><B>Fig. 4 </B>GridSearchCV for min sample leaf in Decision Tree</center>

* The accuracy of training set and validation set both goes down as min samples leaf increases.



From the two graphs above , we can conclude that the accuracy of validation(test) set is maximized when $max\_depth=40$ and $min\_samples\_leaf=1$. 

Actually, considering the combination of two parameters at the same time, we can still have the same conclusion that the best accuracy is $0.84$. 



#### (2) Training and validation demo

From part (1), we can select the final model to be a decision tree with $max\_depth=40$ and $min\_samples\_leaf=1$ . Splitting the raw dataset into training set and validation set, we can fit the model again and obtain the following results.



**Estimation of classification error L**

| training error | validation error |
| :------------: | :--------------: |
|      0.0       |      0.0925      |



### b. LeNet

After training for $80$ epochs, we can plot the following learning curve including loss and accuracy of training set and validation set.

<img src="https://github.com/VitoDH/stat535_project/raw/master/img/learning_curve.png" style="zoom:60%" />

<center><B>Fig. 5 </B>Learning Curve</center>

* Training loss decreases as the number of epochs increases. However, the validation loss first decreases but goes up after $10$ epochs, which indicates minor overfitting.
* Both training and validation accuracy increase as the epochs increases and become steady after 40 epochs.



**Estimation of classification error L**

| training error | validation error |
| :------------: | :--------------: |
|     0.0049     |      0.056       |



---

## Reference

[1]Breiman L, Friedman J, Olshen R, etal. Classification and Regression Trees [M]. New York: Chapman
& Hall, 1984.

[2] Xavier Glorot, Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks [J]. PMLR, 2010.

[3]Y. LeCun, Y. Bengio. Convolutional networks for images, speech, and time-series [ J]. The Handbook of Brain Theory and Neural Networks. MIT Press, 1995.