# Automatic shaping and decomposition of reward functions

## Intro

### Known Unknowns (before reading)

- Restructure the reward function
- Shaped reward function
- Decomposition
- Multieffector problems
- The potential funciton, option set, state abstraction function
- Near-optimal behavior, suboptimal, shaping rewards
- Hierarchical RL
- Shaping function, reward decomposition
- Qualitative Nature $\rightarrow$ Quantitative Nature

### Drawback & Solution

- Drawback: the feedback is very delayed, leading to slow learning
- Solution: to reward intermediate progress towards the goal

### Optimality Preservation

- Basic Challenge: the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do (Sutton & Barto, 2017)
- Condition: if and only if the shaping reward can be written as a difference of some potential function evaluated at the source and destination states

### Transfer Learning & Structural Similarity

- Transfer Method: using shaping rewards as a mechanism for transfer
- Challenge: the ideal potential function, which equals to the true value function, might be quite different from ~~the shaped reward function~~
- Solution: take as input a state abstraction function, which could be represented by a list of the state variables that are considered most relevant to the task

### Multieffector Environment

- Decompose the Agent: the overall agent is decomposed into units and an action is a vector containing a command to each unit
- Decompose the Reward: let each unit know **how much** it is responsible to the observed rewards
- Challenge of reward decomposition: the activities of different units should be fairly independent
- Solution: to decompose a shaped version

## Learning shaping functions

### Background

Reward shaping replaces the original function of an MDP by a shaping reward $\tilde{R}(s,a,s')$, hoping to make the problem easier to solve, and the original MDP $\mathcal{M}$ is modified to $\tilde{\mathcal{M}}$

#### Optimality Preservation Condition

- Sufficiency

  If there exists a potential function $\Phi(s)$ such that

  $$
  \tilde{R}(s,a,s') = R(s,a,s') + \Phi(s') - \Phi(s)
  $$

  then, for any policy $\pi$,

  $$
  \tilde{V}_\pi(s) = V_\pi(s) - \Phi(s)
  $$

- Necessity
  
  If a shaped reward does not correspond to a potential, then, there will be some set of transition probabilities for which optimal policies in $\tilde{\mathcal{M}}$ are suboptimal in $\mathcal{M}$

#### When and how it speeds up learning?

- The ideal shaping function: when $\Phi = V$, the value function of the shaped MDP is identically 0, which is quite easy to learn

- Reduce the exploration

  - No Exploration (Greedy, $h=1$)
  - Random Exploration ($\epsilon$-greedy, $h=1+\epsilon|\mathcal{S}|$)
  - **Shaping in terms of the horizon** (only explore the portion of the state space that is visible within the horizon of states occurring on an optimal trajectory)
  - Full Exploration (DP)

### New Approach

#### Notations

- $\mathcal{M}$: the original MDP
- $z$: a function that maps each state to an abstract state, with a set of state $s$ maps to the abstract state $z$
- $O$: a set of temporally abstract options
  - $o \in O$: consists of a policy $\pi_o$ and a termination set $G_o \sub S$
- $P(s'|s,o)$: given an option $o$ (given $\pi_o$ and $G_o$), (the original MDP) doing actions according to $\pi_o$ starting at state $s$ until a termination state in $G_o$ reached, the probability of this termination state
- $R(s, o)$: the expected total reward until $s' \in G_o$ is reached
- $C_x(w)$: the number of times some state or abstract state $x$ occurs along trajectory $w$
- $w_s$: weight of $s$ in $z$, $w_s = \frac{E_\mathcal{P}[C_s ]}{E_\mathcal{P}[C_z ]}$, which is propotional to its expected frequency of occurence

#### The abstract MDP

Given an MDP $M = (S, A, P, R, D)$, state abstraction $z$, and option set $O$, the the abstract MDP $\bar{M} = (\bar{S},\bar{A},\bar{P},\bar{R},\bar{D})$, where

- $\bar{S} = z(S)$
- $\bar{A} = O$
- $\bar{P}(z,o,z') = \sum_{s \in z}w_s\sum_{s' \in z'}P(s'|s,o)$
- $\bar{R}(z,o) = \sum_{s \in z}w_sR(s,o)$
- $\bar{d}(z) = \sum_{s \in z}d(s)$

##### The purpose of abstract MDP

To approximate the true value function by solving a simpler abstract problem

##### Why is it called abstract?

- State Merging: serveral states in the original MDP may be the same abstract state in the abstract MDP, that is the abstract agent "see" the environment in a more "blurred vision"
- Action Encapsulation: one transition of the abstract MDP may contains many trainsitions of the original MDP, that is the abstract agent take actions and get reward faster (an abstract greedy snake may take several steps and get several rewards upon taking one abstract action)

##### What will the abstract MDP learn compared to the original MDP?

Take the maze problem as example

- The original MDP
  - what's the best next coordinate? ($\pi$)
  - what's the value of each coordinate? ($V_\pi(s)$)
- The abstract MDP
  - what's the best next block? ($\pi_o$)
  - what's the value of each block? ($V_o(z)$)
  - if I have walked into the range of block-2, on which coord should I pause and compute the accumulated reward? ($G_o$)

#### Algorithm: Potential Function Learner

- Input
  - $z$: state abstract function
  - $O$: a set of options
  - $T$: a nonnegative integer

- Initialization
  - $\hat{P}$: the transition probability distribution, namely the policy
  - $\hat{R}$: the reward function, namely the value

- The option is the abstract action
  - Option: An option is an action in abstract MDP, which contains the original MDP's policy and the set of termination states
  - Sample Option: Randomly
  - Execute Option: The agent should follow the policy in the sampled option until it reach a termination state
  - Reward of Option: The total reward received in the execution of the option

- Update: update policy and reward of the abstract MDP

##### Update

- The specific update method is not specified in this algorithm, maybe SARSA, Q, DQN etc.
- The tuple $(z(s), o, r, z(s'))$ is like the $(s, a, r, s')$ in original MDP, which contains all the info needed for update

## Learning Reward Decomposition
