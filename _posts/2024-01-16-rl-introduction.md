---
layout: post
comments: false
title: "Reinforcement Learning"
date: 2024-01-19 01:09:00
tags: reinforcement-learning
toc: true
---

🔄This page will contain a combination of my notes and learning from the Reinforcement Learning Specialization from the University of Alberta. Additionally it will contain personal code and visualizations to better understand certain concepts. 

# Course 1️⃣ - Fundamentals of RL 

- Decision making under uncertainty can be formualized by the $k$-armed bandit problem

The __value__ of an action $a$ is the __expected reward__ when that action is taken. 

$$
\begin{align*}
q_*(a) &\dot{=} \mathbb{E}[ R_t | A_t = a ] \qquad \forall a \in \{1, \dots, k\} \\
       &= \sum_r \underbrace{p(r | a)}_{\text{prob. of observing reward} \; r}r
\end{align*}
$$ 

The goal is to __maximize__ the __expected reward__.

$$
\text{argmax}_a q_*(a)
$$

- Note that $q_*(a)$ is not known, so we need to estimate it
- We can estimate action values $q(a)$ using the sample-average method

#### Sample-Average Method

- Can be used to estimate action values $Q(a)$
$$
\begin{align*}
    Q_t(a) \dot{=} \frac{\text{sum of rewards when} \; a \; \text{taken prior to} \; t}{\text{number of times} \; a \; \text{taken prior to} \; t} = \frac{\sum_{i=1}^{t-1}R_i}{t-1} = \frac{1}{t-1} \sum_{i=1}^{t-1}R_i
\end{align*}
$$

Choosing _greedy action selection_ means choosing the action with the currently largest value estimate $a_G = \text{argmax} Q(a)$. Essentially, $Q(a)$ is the expected cumulative future reward.

#### Incremental Implementation

- Describe how action values can be estimated incrementally using the sample-average method
- Identify how the incremental update rule is an instance of a more general update rule

__Incremental update rule__:

The sample average method in recursive matter is: 

$$
\begin{align*}
    Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_i \\ 
            &= \frac{1}{n} (R_n + \sum_{i=1}^{n-1} R_i) \\ 
            &= \frac{1}{n} (R_n + (n - 1) \underbrace{\frac{1}{n-1}\sum_{i=1}^{n-1}R_i}_{\text{current value estimate} \; Q_n(a)}) \\
            &= \frac{1}{n} (R_n + (n - 1)Q_n) \\
            &= \frac{1}{n} (R_n + nQ_n - Q_n) \\
            &= Q_n + \frac{1}{n}(R_n - Q_n) \\            
    \underbrace{\color{Violet}{Q_{n+1}}}_{NewEstimate} 
            &= \underbrace{\color{RoyalBlue}{Q_n}}_{CurrentEstimate} \color{RoyalBlue}{+} 
               \underbrace{\color{RoyalBlue}{\alpha_n}}_{StepSize}\color{RoyalBlue}{(}
               \underbrace{\color{RoyalBlue}{R_n}}_{Target} \color{RoyalBlue}{-} 
               \underbrace{\color{RoyalBlue}{Q_n}}_{CurrentEstimate}\color{RoyalBlue}{)}
\end{align*}
$$

- $R_n$ is the new experience or immediate reward respectively

__Decaying past rewards__:

$$
\begin{align*}
    \color{Violet}{Q_{n+1}} 
            &= Q_n + \alpha_n \Big( R_n - Q_n \Big) \\
            &= \alpha R_n + Q_n - \alpha Q_n \\ 
            &= \alpha R_n + (1 - \alpha)Q_n \\ 
            &= \alpha R_n + (1 - \alpha)\Big[ \alpha R_{n-1} + (1 - \alpha)Q_{n-1} \Big] \\ 
            &= \alpha R_n + (1 - \alpha) \alpha R_{n-1} + (1 - \alpha)^2Q_{n-1} \\ 
            &= \dots \\ 
            &= \alpha R_n + (1 - \alpha) \alpha R_{n-1} + (1 - \alpha)^2\alpha R_{n-2} + \cdots + (1 - \alpha)^{n-1}\alpha R_{1} + (1 - \alpha)^n Q_1 \\
            &= \color{RoyalBlue}{(1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha (1 - \alpha)^{n-i}R_i}
\end{align*}
$$

#### Exploration vs Exploitation

- __Exploration__ - improve knowledge for _long-term_ benefit
- __Exploitation__ - exploit knowledge for _short-term_ benefit 

__Epsilon-Greedy Action Seleciton__: 

A method to choose between eploration and exploitation.

$$
\begin{align*}
    A_t \leftarrow \left \{
    \begin{array}{ll}
        \text{argmax}_a Q_t(a) & \text{with probability} \; 1 - \epsilon \\
        a \sim Uniform(\{a_1, \dots, a_k \}) & \text{with probability} \; \epsilon 
    \end{array}
\right.
\end{align*}
$$


## GridWorld Example

Let's take a $10 \times 10$ gridworld as an example environment. The agent $\color{RoyalBlue}{A}$ starts in the upper left corner (position $(0,0)$). The goal state $\color{ForestGreen}{G}$ is in $(5,5)$. All $\color{Red}{\text{Red}}$ states give the agent a negative reward of $-1$. All other states, except the goal state, give the agent a neutral reward of $0$. The goal state gives the agent a positive reward of $+1$. Note that $\color{grey}{\text{grey}}$ states are walls, thus, positions onto which the agent can't walk. 

![gridworld10x10]({{ site.baseurl }}/assets/images/gridworld_10x10.png)

### Random Action Selection 

The most trivial and dumb way for the agent $\color{RoyalBlue}{A}$ to reach the goal state $\color{ForestGreen}{G}$ is to always choose a random action. Performing $5$ episodes this leads to the following distribution of rewards per action. The maximal number of timesteps has been set to $10,000$. 

![rdmActionSelection]({{ site.baseurl }}/assets/images/rdm_action_selection__violin_plot.png)

Since the agent $\color{RoyalBlue}{A}$ isn't learning anything the only patterns we can observe is that there are basically never positive rewards for the action _up_ ($0$) and _down_ ($2$).

The following shows the number of timesteps needed to reach the goal state $\color{ForestGreen}{G}$ as well as the cumulative reward `Sum(R)` obtained over all these timesteps. 

```
Episode: 0 -> Successfully finished after 179 steps. Sum(R): -9
Episode: 1 -> Successfully finished after 35  steps. Sum(R): -1
Episode: 2 -> Successfully finished after 327 steps. Sum(R): -17
Episode: 3 -> Successfully finished after 604 steps. Sum(R): -17
Episode: 4 -> Successfully finished after 442 steps. Sum(R): -16
Episode: 5 -> Successfully finished after 396 steps. Sum(R): -1
Episode: 6 -> Successfully finished after 393 steps. Sum(R): -12
Episode: 7 -> Successfully finished after 260 steps. Sum(R): -1
Episode: 8 -> Successfully finished after 265 steps. Sum(R): -2
Episode: 9 -> Successfully finished after 126 steps. Sum(R): -1
```

Every episode finished before the maximal number of timesteps has been reached. Thus, even though choosing randomly and not _learning_ anything, the agent essentially finds the goal state when walking around long enough. 

> Using _random action selection_, it took the agent on average $303 \pm 158$ steps to reach the goal when with an average cumulative reward of $-30.80 \pm 27.54$. 

### Greedy Action Selection

The most trivial method to let the agent _learn_ 

- [ ] Add violin plot with x axis (Action) and y axis (Reward distribution)
- [ ] Show different plots for each episode
- [ ] Show a lineplot with x axis (Steps) and y axis (Reward). First for a single run (episode). Run with a specific random seed.
  - [ ] Then run with another 2nd random seed
  - [ ] Then run with another 3rd random seed
  - [ ] Average the reward over the 3 runs (check [this lecture](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning/lecture/tHDck/what-is-the-trade-off)). y axis is the average reward
  - [ ] Then do for 30 runs  
  - [ ] Then for 100 runs 
  - [ ] Then for 2000 runs 
    - The result will say "What this result says is that this way of behaving obtains this much reward in expectation across possible stochastic outcomes"
  - [ ] Now plot the average rewrd for different $\epsilon$ values (e.g. $0, 0.1, 0.01$)


# ⚙️ Elements of RL 

- __Policy__:
  - > defines the agents way of behaving at a given time
  - mapping of perceived states of the env to actions to be taken when in those states
  - is alone sufficient to determine behavior 
  - stochastic
  - specifies probabilities for each action
- __Reward__ signal: 
  - > goal of the RL problem
  - > what is good in an _immediate_ sense
  - determine the immediate, intrinsic desirability of environmental states
  - pleasure or pain
  - are primary
  - a number send by the env to the RL agent at each time step
  - agent's solve objective is to maximize the total reward it receives over the long run
  - defines what are the good and bad events
- __Value function__:
  - > what is good in the _long_ run
  - __Value__ of a state:
    - > total amount of reward an agent can expect to accumulate over the future, starting from that state
    - indicate the _long-term_ desirability of states
    - farsighted pleasing or displeasing
    - values are predictions of rewards
    - are secondary
    - w/o rewards there could be no values
    - only purpose of estimating values is to achieve more reward
    - action choices are made based on values
- __Model__ of the environment:
  - > mimics the behavor of the env
  - > allows inferences to be made about how the env will behave

The following lists some other elements.

- __Planning__:
  - > deciding on a course of action (by considering possible future situations before they are actually experienced)
- __Model-based__ methods:
  - > methods for solving RL problems that use models and planning
- __Model-free__ methods: 
  - > methods that are explicitly trial-and-error learners
  - opposite of planning 


# Value Iteration vs Policy Iteration


The following shows the pseudocode for _policy iteration_ using iterative policy evaluation for $v^*$. 

$$
\begin{align}

& \text{1. Initialization}  \\
& \qquad v(s) \in \mathbb{R} \; \text{and} \; \pi(s) \in \mathcal{A}(s) \; \text{arbitrarily for all} \; s \in \mathcal{S} \\
\\
& \text{2. Policy Evaluation} \\
& \qquad \text{Repeat} \\
& \qquad \qquad \Delta \leftarrow 0 \\
& \qquad \qquad \text{For each} \; s \in \mathcal{S}: \\ 
& \qquad \qquad \qquad temp \leftarrow v(s) \\ 
& \qquad \qquad \qquad v(s) \leftarrow \sum_{s'} p(s' | s, \pi(s)) \Big[ r(s, \pi(s), s') + \gamma v(s') \Big] \\ 
& \qquad \qquad \qquad \Delta \leftarrow \max( \Delta, | temp - v(s) | ) \\
& \qquad \text{until} \; \Delta < \theta \qquad (\text{a small positive number}) \\
\\
& \text{3. Policy Improvement} \\ 
& \qquad policyStable \leftarrow true \\
& \qquad \text{For each} \; s \in \mathcal{S}: \\ 
& \qquad \qquad temp \leftarrow \pi(s) \\ 
& \qquad \qquad \pi(s) \leftarrow \text{argmax}_a \sum_s p(s' | s,a) \Big[ r(s,a,s') + \gamma v(s') \Big] \\ 
& \qquad \qquad temp \neq \pi(s), \; \text{then} \; policyStable \leftarrow false \\ 
& \qquad \text{If} \; policyStable, \; \text{then stop and return} \; v \; \text{and} \; \pi; \; \text{else go to} \; 2

\end{align}
$$

The following shows the pseudocode for _value iteration_. 

$$
\begin{align}
& \text{Initialize array} \; v \; \text{arbitrarily (e.g., } \; v(s) = 0 \; \text{for all} \; s \in \mathcal{S}^+)  \\
\\
& \text{Repeat} \\
& \qquad \Delta \leftarrow 0 \\
& \qquad \text{For each} \; s \in \mathcal{S}: \\ 
& \qquad \qquad temp \leftarrow v(s) \\ 
& \qquad \qquad v(s) \leftarrow \color{RoyalBlue}{\max_a} \sum_{s'} p(s' | s, \color{RoyalBlue}{a}) \Big[ r(s, \color{RoyalBlue}{a}, s') + \gamma v(s') \Big] \\ 
& \qquad \qquad \Delta \leftarrow \max( \Delta, | temp - v(s) | ) \\
& \qquad \text{until} \; \Delta < \theta \qquad (\text{a small positive number}) \\
\\
& \text{Output a deterministic policy} \; \pi, \; \text{such that} \\
& \qquad \pi(s) = \text{argmax}_a \sum_{s'} p(s' | s, \pi(s)) \Big[ r(s, \pi(s), s') + \gamma v(s') \Big]
\end{align}
$$

A major difference between policy iteration and value iteration is in the update of the value function. Value iteration updates the value of the current state $v(s)$ with the $\color{RoyalBlue}{\max}$ over all actions $a_i$. Policy iteration on the other hand updates $v(s)$ by taking the sum over all next states. Also, value iteration doesn't use any policy $\pi$ to update $v(s)$. 

# 📚 Resources

- [Reinforcement Learning - An Introduction, 2nd Edition, Sutton & Barto](https://mitpress.mit.edu/9780262039246/reinforcement-learning/)
- [Reinforcement Learning Specialization - University of Alberta, Coursera](https://www.coursera.org/specializations/reinforcement-learning)
- [`reinforce-py`](https://github.com/PeeteKeesel/reinforce-py)

