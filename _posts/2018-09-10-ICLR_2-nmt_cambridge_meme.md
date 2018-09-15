---
layout: post
title: Neural Machine Translation of the Cambridge meme
date: 2018-09-10 00:00:00 +0100
description: >
  State-of-the-art NMT systems can't cope with even moderate amounts of noise.
  It is time to call for more robust NMT training.
  <br>
  ("Synthetic and natural noise both break Neural Machine Translation", Belinkov et al., 2018).
img: 2018-09-10-ICLR_2-nmt_cambridge_meme-thumbnail.png
fig-caption: Google Translate example
tags: [ICLR 2018, Neural Machine Translation, Noise, NLP, ICLR]
---


<br>

---



#### ICLR 2018 - 2nd article

*In this series, we explore the 2018 edition of the International Conference
on Learning Representations. Each oral paper is analyzed and
commented in an accessible way.*

*This article is based on the paper*
[Synthetic and natural noise both break Neural Machine Translation](https://openreview.net/pdf?id=BJ8vJebC-)
*by Yonatan Belinkov and Yonatan Bisk.*



---


<br>


If you are reading these lines, there is a good chance that you have been
wandering the Internet for quite some time now. This means that there is also
a good chance that you've stumbled upon the *Cambridge research* meme:


<div class="centeredquote">
Aoccdrnig to a rscheearch at Cmabrigde Uinervtisy, it deosn't mttaer in waht
oredr the ltteers in a wrod are, the olny iprmoetnt tihng is taht the frist and
lsat ltteer be at the rghit pclae.
</div>


There is even a greater chance that you have, for more or less legitimate
reasons, used Google Translate (or even better, its European equivalent, DeepL).
Of course, those automated translation systems have improved a lot in the past
few years; what was laughable at a couple years ago is now quite reasonable for
a number of languages. For simple texts, the translation even feels quite
natural, although not idiomatic (try translating "Il tombe des cordes.", which
is the usual French equivalent of "It's raining cats and dogs.").


Going back to the meme, you would probably have no trouble translating it to
whatever language you are familiar with. But how do automated translation
systems cope with this difficult situation? Let's try translating a French
version into English:


| Source | Sentence |
|--------|--------|
| Original <br>French <br> scrambled | Sleon un cehcruher de l’Uvinertisé de Cmabrigde, l’odrre des ltteers dnas un mot n’a pas d’ipmrotncae, la suele coshe ipmrotnate est que la pmeirère et la drenèire lteetrs sinoet à la bnnoe pclae. |
| Human translation | According to a researcher at Cambridge University, it doesn't matter in what order the letters in a word are, the only important thing is that the first and last letters are in the right place. |
| Google Translate | According to an opinion of the Uvinised of Cmabrigde, the name of the ltteers in one word does not have a name, but the name of the message is that the letter and the letter are very brief at the same time. |
| DeepL | According to a cehcruher of the Uvinertized of Cmabrigde, the odd of the ltteers in a word has no ipmrotncae, the only coshe ipmrotnate is that the pmeirere and the drenere lteetrs sinoetrs in the bnnoe pclae. |



This is quite unconvincing, to say the least. In other words, Google Translation
and DeepL are both very sensitive to this type of noise, whereas humans
seem impressively robust to it. This is just a symptom of the frailty of
automated translation systems:

> While typos and noise are not new to NLP, our systems are rarely trained to
explicitly address them, as we instead hope that the relevant noise will occur
in the training data.


It turns out that this kind of wishful thinking leaves **Neural Machine
Translation (NMT) systems very
brittle**, with their performance dropping quickly in presence of noise. How can
we evaluate this lack of robustness to noise, and what can be done to improve
our models? Let's jump into the paper.


<br><br>



## Evaluating the robustness of current models

We would like to evaluate the performance of NMT models, and how it is affected
by noise. To do that we will, unsurprisingly, have to answer 3 questions:
* Which performance metric?
* Which NMT models?
* Which kind of noise?

<br>

### The performance metric: BLEU

**[BLEU](https://en.wikipedia.org/wiki/BLEU) (bilingual evaluation
understudy)** is a popular metric to evaluate the similarity between the
machine translation and a professional human translation. The idea behind it is
fairly simple: it is a modification of precision, computed on candidate
sentences (by comparison with reference sentences), and then averaged to produce
a corpus-level score of the translation's quality.

Let's first remember why we don't use precision as metric for this task.
Consider a sentence $s$ to be translated, with human reference translation $r$
and candidate translation $c$. Then precision is a score between 0 and 1
computed as:

$$
  \textrm{Precision}(c, r) =
  \dfrac{\sum_{w \in W(c)} m_w(c) \cdot \mathbb{1}_{w \in W(r)}}
    {\sum_{w \in W(c)} m_w(c)}
$$

with $W(c)$ the set of unique words that appear in $c$, and $m_w(c)$ the number
of occurences of $w$ in $c$. Precision essentially measures the proportion of
words of the candidate translation that actually appear in the reference
translation. To see why this is not sufficient, have a look at the table below.
Usually precision is considered coupled with recall to overcome such issues;
we don't, because recall can be artificially inflated when considering multiple
references (see [here](https://en.wikipedia.org/wiki/BLEU)).

| Reference | Candidate | Precision | BLEU (bigram) |
|------|------|------|------|
| "I like trains a lot" | "I like" | $\dfrac{1+1}{2} = 1$ | $\dfrac{1}{1} \cdot e^{1 - 2/5} = 0.55$ |
| "I like trains a lot" | "a a a a a" | $\dfrac{5}{5} = 1$ | $\dfrac{0}{4} \cdot 1 = 0$ |
| "I like trains a lot" | "I like trains a a" | $\dfrac{5}{5} = 1$ | $\dfrac{3}{4} \cdot 1 = 0.75$ |
| "It is raining cats and dogs" | "It is pouring" | $\dfrac{2}{3} = 0.66$ | $\dfrac{1}{2} \cdot e^{1 - 3/6} = 0.30$ |

To solve those issues, the BLEU score brings a couple of modifications to the
precision:
1. Consider the n-grams $G(c)$ instead of the words $W(c)$. <br>This helps with
the fluency of the translation: with words only, "trains lot I a like" gets the
same score as "I like trains a lot".

2. Replace $\ \ \textrm{ }m_w(c) \cdot 1_{w \in W(r)}\ \ \textrm{ }$ by $\ \
\textrm{ }\min (m_w(c), m_w(r)) \cdot {1\_{w \in W(r)}}$. <br>This basically
says that, in the second example, only the first "a" will be counted as correct,
since "a" appears only once in the reference.

3. Multiply by a factor $\ \ \min\left(1, \exp \left(1 - \frac{\textrm{length of reference
corpus}} {\textrm{length of candidate corpus}}\right)\right)$. <br>Indeed, even with the
previous modifications, the constructed score still favors short translations
(see the first example). We therefore penalize translations that are shorter on
average than the reference.

<br>

As illustrated in the last row of the table, even with these modifications the
final score is not ideal, since a perfectly natural candidate translation gets
a low score. It is, however, still much better than precision in a wide range
of situations. In the end, the BLEU score takes the following form
for our example sentences:

$$
  \textrm{BLEU}(c, r) =
  \dfrac{\sum_{g \in G(c)} \min (m_g(r), m_g(c)) \cdot \mathbb{1}_{g \in G(r)}}
    {\sum_{g \in G(c)} m_g(c)}
  \cdot \min\left(1, \exp \left({1 -
  \frac{\textrm{length}(r)}{\textrm{length}(c)}}\right)\right)
$$


<br>


### The NMT models: **Nematus** and **char2char**

This will be much quicker. We want to see how state-of-the-art models
with different architectures (and especially different accesses to words,
sub-word units or characters) are able to cope with noise. The authors ran their
experiments on three distinct models:

* [**char2char**](https://arxiv.org/abs/1610.03017) (Lee et al., 2017) - a
sequence-to-sequence model with attention, trained on characters to characters
([implementation](https://github.com/nyu-dl/dl4mt-c2c)).


* [**Nematus**](http://aclweb.org/anthology/E17-3017) (Sennrich et al., 2017) -
a competition-winning sequence-to-sequence model with some architectural
modifications that enable operating on sub-word units
([implementation](http://data.statmt.org/rsennrich/wmt16_systems/)).

* [**charCNN**](https://arxiv.org/abs/1508.06615) (Kim et al., 2015) - an
attentional sequence-to-sequence model based on a character convolutional
neural network. To quote the authors, "this model retains the notion of a word
but learns a character-dependent representation of words", and "performs well on
morphologically rich languages"
([implementation](https://github.com/harvardnlp/seq2seq-attn)).


<br>


### The noise types: Nat, Key, Swap, Mid, Rand

We finally need to define a number of noise types that we will use to perturb
the models.

| Source | Example | Description |
|--------|--------|--------|
| **Original** | weather | Original correct word |
| **Nat** | whether | Natural noise, e.g. spelling mistake |
| **Key** | qeather | Replace a letter with an adjacent key (QWERTY keyboard) |
| **Swap** | wetaher | Swap two letters except the first and last ones |
| **Mid** | whaeter | Scramble the letters except the first and last ones |
| **Rand** | raewhte | Scramble all letters |





<br>

---

<br>


If you asked people in the deep learning community what their favorite optimization
algorithm is, you'd probably see [Adam](https://arxiv.org/pdf/1412.6980.pdf)
in the top 2 (with Stochastic gradient descent + momentum),
far ahead of alternatives like Adagrad or RMSProp (we'll come to
these in a minute). <b>Adam has become very popular in deep
learning since it was proposed by Kingma & Ba in December 2014</b>.
The reasons are easy to
understand: it exhibits impressive performance, uses many important ideas of
previous works, comes with predefined settings that already work very well,
and does not require careful hyper-parameter tuning.
*Better performance with less work, what more do you want?*

Naturally, great power comes with great responsibilities, even for optimization
algorithms like Adam. At the very least, we expect some theoretical guarantees
on the convergence properties, to make sure that the algorithm is actually doing
its job when required. This is a necessary sanity check, confirming by
the maths that we can legitimately have faith in its correctness.

The convergence proof was provided in the original Adam paper.
However, our authors show in their paper that **this original proof is flawed,
and Adam does not correctly converge in all problems**, as illustrated by this
theorem:

> **Theorem 3.** [With mild assumptions on the parameters], there is a
stochastic convex optimization problem for which Adam does not converge to the
optimal solution.

How did we reach this point, where the most widely used
optimization algorithm in deep learning does not converge in some simple
convex problems? Let's go back some years, and see what lead us there.



<br><br>



## Gradient descent and adaptive methods

To facilitate analysis, let's define our online optimization framework.
At each time step $t$:
* The algorithm picks a
point $x_t$ (weights of the model) in the feasible set $\mathcal{F}$.
* We then
get access to the next mini-batch, with the associated loss function $f_t$,
and we incur the loss $f_t(x_t)$.

Usually, our goal is to find an optimal
parameter $x$ such that the loss $f(x)$ on the entire training set is minimal
(intermediate loss functions $f_t$ are used as stochastic approximations of
$f$). Here, in the (equivalent) online setup, the goal
is to **minimize the total regret at time $T$**:

$$
R_T = \sum_{i=1}^T f_t(x_t) - \min_{x \in
\mathcal{F}} (\sum_{i=1}^T f_t(x))
$$

A large number of optimization algorithms were proposed to achieve this goal.
For a detailed and intuitive overview (and introduction to gradient descent),
I recommend this excellent
[blog post](http://ruder.io/optimizing-gradient-descent/index.html) by Sebastian
Ruder. We will just quickly go over a couple major steps that lead us to Adam.

1. **Mini-batch/Stochastic gradient descent** is the first variant of gradient
descent that actually works decently well in deep learning. In our online setup,
it is reframed as *online gradient descent*, where the points $x_t$ are updated
by moving in the opposite direction of the gradient $g_t = \nabla f_t(x_t)$,
computed on the available mini-batch. The update size is determined by a
learning rate $\alpha_t$ (typically $\alpha/\sqrt{t}$ for some $\alpha$),
and the result is projected back to the feasible set thanks to the projection
operator $\Pi_{\mathcal{F}}$. <br>
We thus get the **update rule of SGD:
$\ x_{t+1} = \Pi_\mathcal{F}(x_t - \alpha_t g_t)$**.

Although the convergence of SGD requires a decreasing learning rate, choosing
an adequate learning rate decrease schedule can be painful. Aggressive decays,
such as $\alpha/\sqrt{t}$, or small learning rates, yield very slow convergence
and mediocre performance. On the other hand, gentle decays and high learning
rates yield very unstable training, even divergence sometimes. To overcome these
issues, **adaptive methods** have been developed around the key idea that **each
weight, that is each coordinate of $x_t$, should be updated using its own
learning rate**, automatically computed based on the knowledge of past updates.
This way, parameters that are frequently updated take only small steps (to avoid
divergence), while parameters that are rarely updated take rather huge steps
(to speed up convergence). This is summed up in the generic adaptive framework
below:

![Adaptive methods]({{site.baseurl}}/assets/img/2018-08-19-adaptive-methods.png){: .center-image}

We call $\alpha_t$ the step size, $\alpha_t/\sqrt{V_t}$ the learning rate,
and restrict ourselves to diagonal variants $V_t = \text{diag}(v_t)$.
This framework essentially allows us to compute a different learning rate for
each weight, by rescaling the step size (common to all weights) with a function
of past gradients (of the loss function with respect to the considered weight,
thus *weight-specific*). Using this framework, we can follow the evolution of
adaptive methods over time:

1. **([SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)) Stochastic gradient descent**
   <center>$\phi_t(g_1, ..., g_t) = g_t\text{  }$ and
   $\text{  }\psi_t(g_1, ..., g_t) = \mathbb{I}$</center>
   SGD is also an adaptive method, with a specific strategy of not adapting at
   all: it forgets everything about the past of each weight, and relies only on
   the current gradient to perform the update, with a step size
   (and learning rate) $\alpha_t =\alpha/\sqrt{t}$, an aggressive decay.<br><br>

2. **([AdaGrad](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)) Adaptive gradient descent**
   <center>$\phi_t(g_1, ..., g_t) = g_t\text{  }$ and
   $\text{  }\psi_t(g_1, ..., g_t) = {\text{diag}(\dfrac{1}{t} \sum_{i=1}^t g_i^2)}$</center>
   With the same step size $\alpha_t = \alpha/\sqrt{t}$, this yields an adaptive
   and much more reasonable learning rate of $\alpha/\sqrt{\sum_i g_{ij}}$ for
   a the $j$-th weight. When the gradients $\{g_{ij}\}$ are sparse, i.e.
   the $j$-th weight is not frequently updated, this **considerably speeds up
   convergence** (updates are still rare, but much bigger than with SGD).<br><br>

3. **([RMSProp](https://www.coursera.org/lecture/neural-networks/rmsprop-divide-the-gradient-by-a-running-average-of-its-recent-magnitude-YQHki))**
   <center>$\phi_t(g_1, ..., g_t) = g_t\text{  }$ and
   $\text{  }\psi_t(g_1, ..., g_t) = {\text{diag}((1-\beta)\sum_{i=1}^t \beta^{t-i} g_i^2)}$</center>
   Despite its huge benefits in some settings, AdaGrad tends to shrink the
   learning rates of frequently updated parameters, very quickly to virtually
   zero. To overcome this issue, RMSProp restricts the average of past gradients
   to a fixed window, instead of the entire past, to avoid shrinking the
   learning rates to virtually zero too quickly. In practice, this is
   implemented by an **exponentially moving average of past gradients**.<br><br>

4. **([Adam](https://arxiv.org/pdf/1412.6980.pdf)) Adaptive momentum estimation**
   <center>$\phi_t(g_1, ..., g_t) = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i^2{\ }\text{  }$ and
   $\text{  }\psi_t(g_1, ..., g_t) = {\text{diag}((1-\beta_2)\sum_{i=1}^t \beta_2^{t-i} g_i^2)}$</center>
   To further speed up the convergence, Adam adds to RMSProp the idea of
   **momentum** (instead of moving in the direction
   of the current gradient only, move in a direction that is a weighted average
   of current gradient and previous update, like a rolling ball's movement is
   is influenced by the both the current slope and its momentum; see
   [Ruder's post](http://ruder.io/optimizing-gradient-descent/index.html)).
   The momentum parameter $\beta_1>0$ considerably
   improves the performance of the algorithm, especially in deep learning.
   Combining momentum with adaptive methods, Adam is very efficient, and
   immensely popular.



<br><br>


## The non-convergence of Adam

So, what's wrong with Adam? It turns out that in the convergence proof given
by Kingma & Ba, a flaw resides in the consideration of a specific quantity,
namely:

$$\ \Gamma_{t+1} = \bigg(\dfrac{\sqrt{V_{t+1}}}{\alpha_{t+1}} -
\dfrac{\sqrt{V_{t}}}{\alpha_{t}}\bigg)$$

Intuitively, $\Gamma_t$ measures the
variations of the inverse of the learning rate. As we saw, convergence requires
a **non-increasing learning rate** (otherwise the algorithm oscillates too much,
or even diverges), which directly translates to $\Gamma_t \succeq O$ (easy to
see with a diagonal $V_t$). This is the case for both SGD ($v_t$ constant,
$\alpha_t$ decreasing) and AdaGrad ($v_t$ non-decreasing,
$\alpha_t$ decreasing). However, it is no longer the case for Adam (nor RMSProp)
because of the exponentially moving average of past gradients.

This is actually a huge issue, meaning that in some cases Adam actually
converges to the worst solution possible. Following the example provided by the
authors, consider $C>2$ and $\mathcal{F} = [-1,1]$, and the following losses:

![ADAM counterexample]({{site.baseurl}}/assets/img/2018-08-19-ADAM_counterexample2.png){: .center-image}

The optimal solution would be $x=-1$. However, choosing $\beta_1 = 0$ and
$\beta_2 = {1}/{(1+C^2)}$, Adam actually converges to the worst possible point
$x = +1$. The intuition for this behavior is the following: although the
algorithm observes a large gradient $C$ every $3$ steps,
**it forgets (scales down) this large gradient too quickly (due to the
exponential weights)** to counterbalance the wrong but most frequent updates.

So you thought Adam was reliable in all convex ("easy") cases? Well, no:
> **Theorem 1.** There is an online convex optimization problem where Adam has
non-zero average regret.

Wait, but this is for a specific choice of parameters. We allowed the choice of
a very small $\beta_2$, and to quote the paper, "this example also provides
intuition for why large $\beta_2$ is advisable [...], and indeed in practice
using large $\beta_2$ helps". So maybe we are safe when using correctly chosen
parameters? Well, no:
> **Theorem 2.** For any constant $\beta_1,\beta_2 \in [0,1)$ such that $\beta_1
< \sqrt{\beta_2}$ [typically satisfied by default settings], there is an online
convex optimization problem where Adam has non-zero average regret.

All right, but this was in online optimization, where one could carefully design
a sequence of loss functions to specifically fool Adam. We should be fine with
stochastic optimization where such a design is impossible, right? Well, no:
> **Theorem 3.** For any constant $\beta_1,\beta_2 \in [0,1)$ such that $\beta_1
< \sqrt{\beta_2}$ [typically satisfied by default settings], there is
a stochastic convex optimization problem for which Adam does not converge to the
optimal solution.


Now this is actually a major concern, since stochastic convex optimization is
actually one of the simplest problems that we expect Adam to be able to solve
(and deep learning consists mostly of non-convex stochastic optimization,
which is reputed much harder). In practice, "fixing Adam" in cases of bad
convergence behavior would typically require using different hyper-parameters
for each dimension which, well, makes you wonder why you're using adaptive
methods in the first place (especially given the very high number of dimensions
in typical deep learning applications).

Another important note is that this analysis remains valid for **any adaptive
method that is based on exponentially weighted moving averages (EWMA) of the
gradients**, including RMSProp, AdaDelta and NAdam, which are therefore also
flawed.

<br><br>


## AMSGrad, a new (and fixed) variant

Now that the maths have spoken and revealed the problem with Adam,
what can be done? EWMA-based algorithms
have actually brought a lot in terms of performance improvement and robustness
to hyper-parameters choice, and it would be quite a pity to simply throw it all
away for just a hole discovered in the proof.

![AMSGrad]({{site.baseurl}}/assets/img/2018-08-19-AMSGRAD.png){: .center-image}

To overcome this issue, the authors propose a new variant, AMSGrad, described
in the algorithm above, that we explain in the next few lines.
Remember that the issue with Adam resides in $\Gamma_t$, which should be
semi-definite positive, while in some cases it is not because
$\frac{v_{t+1, j}}{\alpha_{t+1}} - \frac{v_{t, j}}{\alpha_{t}} < 0$ for some
index $j$. Well, a quick hotfix would be to replace $v_{t+1}$ by $\max(v_t,
v_{t+1})$, which would ensure $\Gamma_t \succeq 0$. But this would bias our
computations of the EWMA...
However we can decouple the two processes by keeping
two running variables: one "true" running variable $v_{t+1}$ used to accurately
compute the EWMA (correctly accounting for the past), and one "used-in-updates"
running variable ${\hat v_{t+1}} = {\max(v_{t+1}, \hat v_t)}$ used to ensure
a non-increasing learning rate (hence the correctness of the algorithm).

Theoretical justifications of AMSGrad back up this claim: for reasonable choices
of parameters, the regret is bounded in $O(\sqrt{T})$. Additionally, for a
specific choice of momentum decay $\beta_{1t} = \beta_1 \lambda^{t-1}$, the
authors prove a bound on the regret that can hugely benefit from sparse
gradients, potentially "considerably better than $O(\sqrt{dT})$ regret of SGD".

Does this translate into good empirical performance? The paper provides the
convergence curves for a synthetic example that is close to the previous
example (with $\mathcal{F} = [-1, 1]$, the optimal solution being $x=-1$):

![Synthetic example]({{site.baseurl}}/assets/img/2018-08-19-synthetic_example.png){: .center-image}

![Convergence curves]({{site.baseurl}}/assets/img/2018-08-19-convergence_curves.png){: .center-image}

<br>

This example shows that AMSGrad works on the synthetic example where Adam fails
to converge to the true solution, which is always a good sign, since it was why
it was designed in the first place. But what about standard convergence
benchmarks on more interesting problems, like logistic regression or neural
networks? Reddi et al. provide graphics that claim for AMSGrad
equal or better performance compared to Adam for logistic regression,
feed-forward neural net (1 hidden layer) on MNIST, and CIFARNET. However, in an
independent
[implementation by Filip Korzeniowski](https://fdlm.github.io/post/amsgrad/)
tested in various settings (you should definitely check his post for extensive
details and comparisons), the experiments do not support any claim of
practical difference between Adam and AMSGrad. We show below the validation
accuracy (what we care about in the end) he gets for both algorithm on CIFAR-10.

![Comparison]({{site.baseurl}}/assets/img/2018-08-19-comparison.png){: .center-image}

<center>Validation accuracy on CIFAR-10 (VGG-16).
<br>Adam in blue, AMSGrad in red.
</center>

<br><br>


## Does it matter?

In my opinion, it is pretty clear that the main interest of the paper lies
in pointing out the flaw in Adam's proof, and how it affects the convergence
behavior in some cases. AMSGrad sounded somewhat more like a hotfix than a real
replacement, providing a sounder algorithm while maintaining Adam's good
performance. AMSGrad will not change the course of stochastic optimization's
(nor deep learning's) history, may it even come to someday replace Adam
in practice. *In the end, it doesn't even matter (that much)* ; a true
"breakthrough" replacement is yet to come.

What is more interesting is the meta-aspect of the proof-checking side of this
paper. It seems very important to me that empirical observations of bad
convergence behavior are finally backed up by theoretical justifications, in
other words, that **someone finally checked the maths**. I find it actually
surprising that despite its "reasonable" length of 4 pages, no one checked the
proof and spotted this mistake, while in the meantime Adam got increasingly
popular and widely used. Note that I barely ever read the proofs myself. I would
tend to link this phenomenon to a deeper trend in the machine/deep learning
community: people (and scientists are people too) get really excited about the
results, the performance and the new opportunities brought by an exponentially
growing field, and we just forget how to do good science along the way. The
reproducibility crisis denounced by researchers like Joëlle Pineau
(especially in reinforcement learning) is one example; the fact that everyone
skips the proofs to go straight to the next exciting paper is another.

<br>
