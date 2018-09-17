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


### Facing the truth - the (in)ability to translate noisy texts

With all the tools in hand, it is now time to experiment and face the truth:
how well are our NMT models able to cope with the different noise types? I'm
definitely too lazy to run all the tests by myself, so I'll steal the paper's
results instead.

![Experiments with noise]({{site.baseurl}}/assets/img/2018-09-10-noise_results.png){: .center-image}

![Performance decrease]({{site.baseurl}}/assets/img/2018-09-10-noise_plots.png){: .center-image}

Whatever the language, noise has a striking effect on performance.
As a quick note, the results presented in the table aren't comparable across
noise types, since the noise intensity is not the same in all cases. For a fair
comparison, have a look at the graphs (German to English): surprisingly enough,
the Rand noise isn't much worse than Swap, despite yielding much more dramatic
changes of the tokens. It is actually *very* surprising, since Swap is strictly
included in Rand. This would mean that the NMT model (especially Nematus)
is basically unable to recover a word with virtually any Swap mistake.
Yet, the authors' conclusion still holds true:

> The important thing to note is that even small amounts of noise lead to
substantial drops in performance. [...] This is true for both natural noise and
all kinds of synthetic noise. [...] The degradation in quality is especially
severe in light of the humans' ability to understand noisy texts.



<br><br>



## Dealing with noise

Now that we know where we are, what can be done to increase the robustness of
our models? The authors propose two natural ideas, and show their effect on
performance:
1. All models presented rely on word structure to build a representation. This
structure is, however, altered to some extent by most of the studied noise
types (Swap, Mid, Rand). The idea is therefore to make some architectural
modifications in order to learn a representation of words that is invariant to
their structure.
2. Training on noisy examples has been regularly reported to increase model
robustness to noise. Let's try that too.


<br>


### Structure invariant representations

As is clear in their architecture, all three of the models under study are
*by design* sensitive to word structure, at a character level (due to
convolutional layers or the sub-word units considered). There is a fair chance
that a model that is *insensitive* to this structure would be more robust to
noises that affect this structure; nothing groundbreaking there.

> Perhaps the simplest such model is to take the average character embedding as
a word representation. This model, referred to as meanChar, first generates a
word representation by averaging character embeddings, and then proceeds
with a word-level encoder similar to the charCNN model.

![meanChar performance]({{site.baseurl}}/assets/img/2018-09-10-meanchar_performance.png){: .center-image}

<center> Performance of meanChar in different settings. By design, Scr is the
same as Vanilla for this model. </center>

As is clear in the table, meanChar turns out to be pretty good for translating
scrambled text (much better than charCNN). Note that, by design, this model
sees no difference between vanilla and scrambled texts; the Scr performance
reported can therefore be compared to the vanilla performance of other models.
However, meanChar is still very sensitive to Nat and Key noise types, that do
not resemble Rand.


#### Personal notes on structure invariant and graph-based representations

To me, this model seems particularly *blunt*. While it is true that averaging
all character embeddings removes the reliance on word structure, it also
discards a huge part of the information brought by individual character. Unless
you are using an embedding dimension that is of the order of the size of your
alphabet, the
signal transmitted to the convolutional layer dismisses *some* information
about the *presence* of individual characters, in addition to throwing away
all information about *order*.
In other words, not only anagrams will get the same representation, but also
some totally different words could by random chance.
One could argue that, with a good encoding
(e.g. simply using the powers of 2, which is basically cheating by pretending
that a full-dimensional one-hot vector can be reduced to a 1-dimensional scalar
value), we could always recover the presences; however I cannot think of one
that would suit the convolutional structure of the next layer.

Have a look at Figures 2 and 3 below, depicting the charCNN vs. meanChar
word representations. I am pretty sure
that a better word representation can be found, discarding only the structure
while preserving the presence (instead of "jeter le bébé avec l'eau du bain",
like we say in French). A first thought is a complete **graph of individual
characters** (see Figure 4), with a graph-convolutional layer that would follow.
This seems to me like a more natural generalization of the previous models:
thinking in terms of graphs where nodes are the characters, previous
representations saw a word as a chain of characters, each of them linked only to
the next one by a directed edge. Discarding structure would simply mean linking
all nodes together by undirected edges.

This could even
be adapted to retain *some weak* information about the structure by **weighting
edges** using the original order (the closer two characters in the original
word, the higher the weight; see Figure 5). These weights will typically play
a role in interaction with regularization, for example $L_2$-regularization on
the coefficients of the convolutional filters (for very low weights, it will
be too expensive to take those into account unless there is a very good reason).
While not completely structure-invariant, this has a potential to make the
model more robust to
structure change, while retaining all information about the presence of
individual characters and not forgetting all about structure like in meanChar.


![Graph chain]({{site.baseurl}}/assets/img/2018-09-10-graph_chain.png){: .center-image}

<center> <strong>Figure 2</strong> - `charCNN` word representation
as a chain of character embeddings. </center>

![Graph chain]({{site.baseurl}}/assets/img/2018-09-10-graph_meanchar.png){: .center-image}

<center> <strong>Figure 3</strong> - `meanChar` word representation as an average
of character embeddings. </center>

![Graph complete]({{site.baseurl}}/assets/img/2018-09-10-graph_complete.png){: .center-image}

<center> <strong>Figure 4</strong> - `graphChar` word
representation as a complete graph of character embeddings. </center>

![Graph weighted]({{site.baseurl}}/assets/img/2018-09-10-graph_weighted.png){: .center-image}

<center> <strong>Figure 5</strong> - `w-graphChar` word
representation as a weighted graph of character embeddings. The weights, here
arbitrary, could reflect the structure to some extent. </center>


<br>


### Training on noisy texts
