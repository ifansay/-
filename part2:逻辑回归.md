# 1.逻辑回归与线性回归的联系与区别
逻辑回归与线性回归都属于广义线性回归模型,其区别与联系从以下几个方面比较：

**分类与回归:** 
回归模型就是预测一个连续变量(如降水量，价格等)。在分类问题中，预测属于某类的概率，可以看成回归问题。这可以说是使用回归算法的分类方法。

**输出:** 
直接使用线性回归的输出作为概率是有问题的，因为其值有可能小于0或者大于1,这是不符合实际情况的，逻辑回归的输出正是[0,1]区间。

**参数估计方法：**
线性回归中使用的是最小化平方误差损失函数，对偏离真实值越远的数据惩罚越严重。这样做会有什么问题呢？假如使用线性回归对{0,1}二分类问题做预测，则一个真值为1的样本，其预测值为50，那么将会对其产生很大的惩罚，这也和实际情况不符合，更大的预测值说明为1的可能性越大，而不应该惩罚的越严重。
逻辑回归使用对数似然函数进行参数估计，使用交叉熵作为损失函数，对预测错误的惩罚是随着输出的增大，逐渐逼近一个常数，这就不存在上述问题了1
也正是因为使用的参数估计的方法不同，线性回归模型更容易受到异常值(outlier)的影响，有可能需要不断变换阈值(threshold).
无异常值的线性回归情况:
蓝线为求得的h(x)，上图中可选阈值为0.5作为判断肿瘤是否是良性。
有异常值的线性回归情况:
这个时候再想有好的预测效果需调整阈值为0.2，才能准确预测。
使用逻辑回归的方法进行分类，就明显对异常值有较好的稳定性。

**参数解释:** 
线性回归中，独立变量的系数解释十分明了，就是保持其他变量不变时，改变单个变量因变量的改变量。
逻辑回归中，自变量系数的解释就要视情况而定了，要看选用的概率分布是什么，如二项式分布，泊松分布等


# 2.逻辑回归的原理 
Sigmoid 函数是所有函数图像为 S-形的函数的统称。Logistic 函数是形如下式定义的函数

σ(x;α)=11+exp(−α⋅x).
此处 α 是函数的参数，它调整函数曲线的形状。当参数 α=1 时，它的函数曲线如下图所示，因而它是一个 Sigmoid 函数。



当 α 增大时，函数曲线在 x 轴方向压缩，函数曲线越接近阶梯函数。反之，当 α 减小时，函数曲线在 x 轴方向拉伸。通常，我们可以直接使用 α=1 的 Logistic 函数，即：

σ(x)=11+exp(−x).
导函数
Logistic 函数的导函数具有很好的形式，具体来说：

σ′(x)====(11+exp(−x))′−1(1+exp(−x))2⋅exp(−x)⋅(−1)11+exp(−x)⋅exp(−x)1+exp(−x)σ(x)(1−σ(x))
模拟概率
由于 Logistic 函数的值域是 (0,1) 且便于求导，它在机器学习领域经常被用来模拟概率。

具体来说，假设二分类模型有判别函数 z=f(x⃗ ;w⃗ )。其表意为：当输出值 z 越大，则 x⃗  代表的样本为正例的概率越大；当输出值 z 越小，则 x⃗  代表大样本为负例的概率越大。此时，考虑到 Logistic 函数的值域，我们可以用 P(x⃗ )=σ(f(x⃗ ;w⃗ )) 来表示 x⃗  代表的样本为正例的概率。同时，由于 Logistic 函数便于求导，只要我们选用了合适的损失函数（例如交叉熵损失函数），我们就可以方便地将梯度下降法运用在求解参数向量 w⃗  之上。

# 3.逻辑回归损失函数推导及优化 
回顾下线性回归的损失函数，由于线性回归是连续的，所以可以使用模型误差的的平方和来定义损失函数。但是逻辑回归不是连续的，自然线性回归损失函数定义的经验就用不上了。不过我们可以用最大似然法来推导出我们的损失函数。

　　　　我们知道，按照第二节二元逻辑回归的定义，假设我们的样本输出是0或者1两类。那么我们有：

　　　　P(y=1|x,θ)=hθ(x)
　　　　P(y=0|x,θ)=1−hθ(x)
　　　　 把这两个式子写成一个式子，就是：

　　　　P(y|x,θ)=hθ(x)y(1−hθ(x))1−y
　　　　其中y的取值只能是0或者1。

　　　　得到了y的概率分布函数表达式，我们就可以用似然函数最大化来求解我们需要的模型系数θ。

　　　　为了方便求解，这里我们用对数似然函数最大化，对数似然函数取反即为我们的损失函数J(θ)。其中：

　　　　似然函数的代数表达式为：

　　　　L(θ)=∏i=1m(hθ(x(i)))y(i)(1−hθ(x(i)))1−y(i)
　　　　其中m为样本的个数。

　　　　对似然函数对数化取反的表达式，即损失函数表达式为：

　　　　J(θ)=−lnL(θ)=−∑i=1m(y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i))))
 　　　　损失函数用矩阵法表达更加简洁：

　　　　J(θ)=−YT∙loghθ(X)−(E−Y)T∙log(E−hθ(X))
　　　　其中E为全1向量,∙为内积。
    对于二元逻辑回归的损失函数极小化，有比较多的方法，最常见的有梯度下降法，坐标轴下降法，等牛顿法等。这里推导出梯度下降法中θ每次迭代的公式。由于代数法推导比较的繁琐，我习惯于用矩阵法来做损失函数的优化过程，这里给出矩阵法推导二元逻辑回归梯度的过程。

　　　　对于J(θ)=−YT∙loghθ(X)−(E−Y)T∙log(E−hθ(X))，我们用J(θ)对θ向量求导可得：

　　　　∂∂θJ(θ)=XT[1hθ(X)⊙hθ(X)⊙(E−hθ(X))⊙(−Y)]+XT[1E−hθ(X)⊙hθ(X)⊙(E−hθ(X))⊙(E−Y)]
　　　　这一步我们用到了向量求导的链式法则，和下面三个基础求导公式的矩阵形式：

　　　　∂∂xlogx=1/x
　　　　∂∂zg(z)=g(z)(1−g(z))(g(z)为sigmoid函数) 

　　　　∂xθ∂θ=x 

　　　　对于刚才的求导公式我们进行化简可得：

　　　　∂∂θJ(θ)=XT(hθ(X)−Y)
　　　　从而在梯度下降法中每一步向量θ的迭代公式如下：

　　　　θ=θ−αXT(hθ(X)−Y)
　　　　其中，α为梯度下降法的步长。

　　　　实践中，我们一般不用操心优化方法，大部分机器学习库都内置了各种逻辑回归的优化方法，不过了解至少一种优化方法还是有必要的。
# 4.正则化与模型评估指标 
辑回归也会面临过拟合问题，所以我们也要考虑正则化。常见的有L1正则化和L2正则化。

　　　　逻辑回归的L1正则化的损失函数表达式如下，相比普通的逻辑回归损失函数，增加了L1的范数做作为惩罚，超参数α作为惩罚系数，调节惩罚项的大小。

　　　　二元逻辑回归的L1正则化损失函数表达式如下：

　　　　J(θ)=−YT∙loghθ(X)−(E−Y)T∙log(E−hθ(X))+α||θ||1
　　　　其中||θ||1为θ的L1范数。

　　　　逻辑回归的L1正则化损失函数的优化方法常用的有坐标轴下降法和最小角回归法。

 

　　　　二元逻辑回归的L2正则化损失函数表达式如下：

　　　　J(θ)=−YT∙loghθ(X)−(E−Y)T∙log(E−hθ(X))+12α||θ||22
　　　　其中||θ||2为θ的L2范数。

　　　　逻辑回归的L2正则化损失函数的优化方法和普通的逻辑回归类似。
# 5.逻辑回归的优缺点 
优点：

1）预测结果是界于0和1之间的概率；

2）可以适用于连续性和类别性自变量；

3）容易使用和解释；



缺点：

1）对模型中自变量多重共线性较为敏感，例如两个高度相关自变量同时放入模型，可能导致较弱的一个自变量回归符号不符合预期，符号被扭转。​需要利用因子分析或者变量聚类分析等手段来选择代表性的自变量，以减少候选变量之间的相关性；

2）预测结果呈“S”型，因此从log(odds)向概率转化的过程是非线性的，在两端随着​log(odds)值的变化，概率变化很小，边际值太小，slope太小，而中间概率的变化很大，很敏感。 导致很多区间的变量变化对目标概率的影响没有区分度，无法确定阀值。
# 6.样本不均衡问题解决办法
1.扩充数据集

2.对数据集进行重采样

3.人造数据

4.改变分类算法

5.尝试其他评价指标
# 7.sklearn参数
penalty : str, ‘l1’ or ‘l2’, default: ‘l2’ 
惩罚项l1或者l2 l1可以使weight稀疏，l2可以使weight均衡，当solvers 为newton-cg’, ‘sag’ and ‘lbfgs’时，只可以是l2 
C : float, default: 1.0 
正则化的强度 
fit_intercept : bool, default: True 
默认为true，此参数为截距，即y=ax+b的b 
intercept_scaling : float, default 1. 
Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic_feature_weight. 
Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased. 
class_weight : dict or ‘balanced’, default: None 
默认是balanced，即{0：1，1：1}， 如果label中0比较重要，我就可以{0：2，1：1}，即代价敏感学习，一般在样本不平衡中使用

solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}, default: ‘liblinear’ 
优化的算法 
数据比较少时，用liblinear是一个比较好的选择。在数据比较多的情况下，sag更快一些 
对于多分类问题，only ‘newton-cg’, and ‘lbfgs’ 可以处理multinomial loss， ‘liblinear’只可以解决ovr 
‘newton-cg’, ‘lbfgs’ and ‘sag’ 只可以解决l2范式

multi_class : str, {‘ovr’, ‘multinomial’}, default: ‘ovr’ 
If the option chosen is ‘ovr’, then a binary problem is fit for each label. Else the loss minimised is the multinomial loss fit across the entire probability distribution. Works only for the ‘newton-cg’, ‘sag’ and ‘lbfgs’ solver.

n_jobs : int, default: 1 
默认是1，此参数为线程数，可以根据个人电脑增加

Attributes
coef_ : array, shape (n_classes, n_features) 
每一维特征的系数，即weight 
intercept_ : array, shape (n_classes,) 
截距，即bias
