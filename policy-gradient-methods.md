# Policy Gradient Methods

To learn a parameterized policy function instead of a parameterized value function:

$$
\pi(a|s,\theta) = \Pr\{A_t=a|S_t=a,\theta_t=\theta\}
$$

where $\theta$ is the policy's parameter vector. A performance measure $J(\theta)$ of the policy is needed, by maximizing which will the policy to be optimized:

$$
\theta_{t+1} = \theta_t + \alpha\nabla\widehat{J(\bold{\theta_t})}
$$

$J(\theta)$ is often defined as the value of the initial state.

## Policy Approximation and its Advantages

### Conditions

- $\pi(a|s,\theta)$ must be differentiable, which means $\nabla_{\theta}\pi(a|s,\theta)$ exists and is always finite.
- To ensure exploration, generally the policy never becomes deterministic, which means $\pi(a|s,\theta)\in(0,1), \forall s,a,\theta$.

### Parameterization: Numerical Preferences

$$
\pi(a|s,\theta) = \frac{\exp(h(s,a,\theta))}{\sum_b\exp(h(s,b,\theta))}
$$

where the preference themselves can be parameterized arbitrarily, for example, linearly

$$
h(s,a,\theta) = \theta^Tx(s,a)
$$

### Advantages

- The approximated policy can approach a deterministic policy, whereas the action value approximation will always have an $\epsilon$ probability of selecting a random action.

- The choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy into the reinforcement learning system.

- The approximated policy can learn to select certain action in any probability, while the value-based method can only learn the best action and to select this action with a relatively fixed probability.

## The Policy Gradient Theorem

Define $J(\theta) \doteq v_{\pi_0}(s_0)$

### What will affect the performance?

- Given the state distribution, the action selection may be the only factor affecting the performance.
- State distribution: the probability of occurrence of each state, namely the importance. This can reflect the probabilities of the transitions between $s_0$ and other states, so that the $v_{\pi_\theta}$ can be calculated through all q values.

### The Performance

$$
\begin{aligned}
v_{\pi}(s_0) &= \sum_{s\in\mathcal{S}}\sum_{k=0}^{\infty}\Pr(s_0 \rightarrow s,k,\pi)\sum_a \pi(a|s,\theta) q_\pi(s,a)
\end{aligned}
$$

where $\Pr(s_0 \rightarrow s,k,\pi)$ means the probability of transitioning from state $s_0$ to $s$ in $k$ steps under policy $\pi$.

### The Gradient

$$
\begin{aligned}
\nabla J(\theta) &= \nabla v_\pi(s_0) \\
&= \sum_s(\sum_{k=0}^{\infty}\Pr(s_0 \rightarrow s,k,\pi))\sum_a \nabla\pi(a|s,\theta) q_\pi(s,a) \\
&\propto \sum_s \mu(s) \sum_a \nabla\pi(a|s,\theta) q_\pi(s,a) \\
\end{aligned}
$$

## REINFORCE

### From Sum to Expectation

The $\mu(s)$ of each state can be reflected from the Monte Carlo simulation. $s$ can be replaced with a random variable $S_t$ since each $s$ is weighted by a respective $\mu(s)$ and $\sum_s mu(s) = 1$.

$$
\nabla J(\theta) = E_\pi[\sum_a q_\pi(S_t,a) \nabla_\theta \pi(a|S_t,\theta)]
$$

The Monte Carlo also wants to sample random actions, so $a$ should also be replaced with a random variable $A_t$. It's easy to see that $\sum_a \pi(a|s,\theta) = 1$, so

$$
\begin{aligned}
\nabla J(\theta) &= E_\pi[\sum_a \pi(a|S_t,\theta)q_\pi(S_t,a)\frac{\nabla_\theta \pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}] \\
&= E_\pi[q_\pi(S_t, A_t) \frac{\nabla_\theta \pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}] \\
&= E_\pi[G_t \frac{\nabla_\theta \pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}]
\end{aligned}
$$

### Intuitive Explanation

- $\nabla_\theta \pi(a|S_t,\theta)$: The direction the most increases the probability of repeating $A_t$ on future visits to state $S_t$.
- $G_t$: The larger the return is, the more the gradient ascent will go.
- $\pi(a|S_t,\theta)$: For seldom selected actions, it's necessary to increase its effect.

## REINFORCE with Baseline

$$
\nabla J(\theta) \propto \sum_s \mu(s) \sum_a \nabla\pi(a|s,\theta) (q_\pi(s,a)-b(s))
$$

where $b(s)$ is a baseline which never varies with $a$, therefore, the expectation of the gradient will stay unchanged, but the variance will be decreased as the baseline will generally vary with $s$.
