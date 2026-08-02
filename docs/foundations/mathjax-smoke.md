# MathJax smoke test

This page is both a rendering check and the first mathematical convention used by later chapters.
For logits $z_i$, softmax assigns token probability

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}.
$$

For an autoregressive sequence of tokens $x_1,\ldots,x_T$, its log probability is

$$
\log p(x_{1:T}) = \sum_{t=1}^{T} \log p(x_t \mid x_{<t}).
$$

Authored documentation uses dollar-sign delimiters. The documentation linter rejects the raw
delimiter forms that are incompatible with the configured rendering path.
