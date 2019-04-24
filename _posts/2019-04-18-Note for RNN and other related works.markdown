---
layout: post
title:  "Note for RNN and other related works"
date:   2019-04-18 18:10:09 +0800
categories: note
---

# LSTM
For general-purpose sequence modeling, LSTM as a special RNN structure has proven stable and powerful for modeling long-range dependencies.

The major innovations of LSTM:
* $$c_t$$ Memory cell, accumulator of the state information
* $$i_t$$ input gate
* $$f_t$$ forget gate
* $$h_t$$ final state
* $$o_t$$ output gate

> $$\circ$$ denotes the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) (entrywise product).

The key equations are shown below:

$$\begin{aligned}
	i_{t} &=\sigma\left(W_{x i} x_{t}+W_{h i} h_{t-1}+W_{c i} \circ c_{t-1}+b_{i}\right) \\
	f_{t} &=\sigma\left(W_{x f} x_{t}+W_{h f} h_{t-1}+W_{c f} \circ c_{t-1}+b_{f}\right) \\
	c_{t} &=f_{t} \circ c_{t-1}+i_{t} \circ \tanh \left(W_{x c} x_{t}+W_{h c} h_{t-1}+b_{c}\right) \\
	o_{t} &=\sigma\left(W_{x o} x_{t}+W_{h o} h_{t-1}+W_{c o} \circ c_{t}+b_{o}\right) \\
	h_{t} &=o_{t} \circ \tanh \left(c_{t}\right) 
\end{aligned}$$

> [LSTMs are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn!](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">
<img src="/asserts/post/2019-04-18-Note for RNN and other related works.markdown/LSTM3-chain.png">
</a>
<a href="https://colah.github.io/posts/2015-08-Understanding-LSTMs/">
<img src="/asserts/post/2019-04-18-Note for RNN and other related works.markdown/LSTM2-notation.png">
</a>
