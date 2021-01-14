# Deep contextualized word representations

## ELMo: Ebedding from Language Model

ELMo representations are computed on top of two-layer biLMs with character convolutions, as a linear function of the internal network states.

### Bidirectional Language Models

#### Forward Language Model

- Joint Distribution

$$
p(t_1,t_2,\cdots,t_N) = \prod_{k=1}^N p(t_k|t_1,t_2,\cdots,t_{k-1})
$$

- Independencies: each token is independent from other tokens given its previously occurred tokens.

#### Backward Language Model

- Joint Distribution

$$
p(t_1,t_2,\cdots,t_N) = \prod_{k=1}^N p(t_k|t_{k+1},t_{k+2},\cdots,t_N)
$$

- Independencies: each token is independent from other tokens given its following tokens.

#### The Learning Target of biLM

- Target: to maximize the log likelihood of two directions

- Mathematical Expression

$$
\arg\max_{\Theta_x, \overrightarrow{\Theta}_{LSTM}, \overleftarrow{\Theta}_{LSTM}, \Theta_s} \sum_{k=1}^N (\log p(t_k|t_1,\cdots,t_{k-1};\Theta_x, \overrightarrow{\Theta}_{LSTM}, \Theta_s) + \log p(t_k|t_{k+1},\cdots,t_N;\Theta_x, \overleftarrow{\Theta}_{LSTM}, \Theta_s))
$$

where $\Theta_x$ is the parameters for token representation, $\overrightarrow{\Theta}_{LSTM}$ is for the forward LSTM, $\overleftarrow{\Theta}_{LSTM}$ is for the backward LSTM, $\Theta_s$ for the softmax.

### ELMo

#### Definition

Task-specific combination of the intermediate layer representations in the biLM

- task-specific: different combination weights for different tasks.
- intermediate layers: not only the final outputs are utilized, but also the hidden outputs.
- combination: the output of each layer is linearly combined to output the final result.

#### Calculation

- $R_k$: the set of the representation of the $k$-th word of different layers, defined as
  $$
  R_k = \{\bold{h}_{k,j}^{LM}|j=0,1,...L\}
  $$
  where L is the number of layers in the biLM, $\bold{h}_{k,j}^{LM}$ is the representation of the $k$-th word in the $j$-th layer.
- $\bold{ELMo}_k$: the ELMo vector of the $k$-th word, defined as the following two different ways
  $$
  \begin{aligned}
  \bold{ELMo}_k &= E(R_k) = \bold{h}_{k,L}^{LM} \\
  \bold{ELMo}_k &= E(R_k; \Theta_{task}) = \gamma^{task}\sum_{j=0}^Ls_j^{task}\bold{h}_{k,j}^{LM}
  \end{aligned}
  $$
where the first definition only utilize the final output. In the second definition, $s^{task}$ are softmax normalized task-specific weights. $s^{task}$ and $\gamma^{task}$ are all trainable.

### Using biLMs for supervised NLP tasks

#### Process Overview

- Run the biLM and record all of the layer representations for each word
- Let the end task model learn a linear combination of these representatiosn

#### Combine ELMo with the Downstream (Supervised) Model

Most supervised NLP models share a common architecture at the lowest layers, so the ELMo can be added to those layers in a unified manner.

Concatenate the ELMo vector with $\bold{x}_k$ and pass the ELMo enhanced representation into the Downstream model, so that:

- the input is ELMo enhanced with contextual information (contextualized)
- $\Theta^{task}$ will be learned, which means an optimal combination of the intermediate layers will be learned.

From another perspective, this is to concatenate a pre-trained ELMo before the untrained supervised model. In some tasks, the ELMo can also be added after the supervised model to enhance the output.

### Pre-trained bidirectional language model architecture

- The biLM provides 3 layers of representations for input tokens, while the traditional word embedding provides only 1 layer of representations.
- The biLM is of another architecture different from the task RNN. The pre-train is task-free.Once pretrained, the biLM can compute representations $R_k$ for any task.
- In order to adapt to different tasks, the result $R_k$ should further be weighted by trainable weights.
- ELMo is a feature-based method, as the weights of biLM is frozen, while another method, fine-tuning can fine-tune the weights of biLM.

### What will be pre-learned?

- Contextualization:
  - Disambiguation
  - POS tagging

- CV vs. NLP
  - In CV, use a pre-trained CNN model to pre-process a image, converting it to a feature map, then use the feature map instead of (or in addition to) the original image as the input of specific CV task.
  - In NLP, use a pre-trained language model to pre-process a sequence of tokens, converting it to a contextualized sequence, then use the contextualized one instead of (or in addition to) the original one as the input of specific NLP task.
  - Each layer of the pre-trained CV model has learned different features (edges, parts, ...)
  - Each layer of the pre-trained language model has also learned different features (disambiguation, POS, ...)
