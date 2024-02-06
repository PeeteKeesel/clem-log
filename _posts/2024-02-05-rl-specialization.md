---
layout: post
comments: false
title: "Reinforcement Learning Specialization"
date: 2024-02-04 12:00:00
tags: reinforcement-learning
# toc: true
---

This page contains summaries and personal notes obtained during my studying for the [Reinforcement Learning Specializations](https://www.coursera.org/specializations/reinforcement-learning) from the [University of Alberta](https://www.ualberta.ca/index.html) on [Coursera](https://www.coursera.org/). Note that some of the content is completely taken from the course and therefore the credit goes to the University of Alberta and Coursera. 

# 1. Course 1️⃣ - Fundamentals of Reinforcement Learning

## 1.1. Week 🕐

### 1.1.1. The K-Armed Bandit Problem
#### 1.1.1.1. Sequential Decision Making with Evaluative Feedback

- __Decision making under uncertainty__ can be formalized by the __k-armed bandit problem__
- Fundamental ideas: __actions__, __rewards__, __value functions__

### 1.1.2. What to Learn? Estimating Action Values
#### 1.1.2.1. Learning Action Values

- __Sample-average method__ can be used to estimate action values
- The __greedy action__ is the action with the highest value estimate

<u>Action-Values</u>

$$
\begin{align*}
q_*(a) &\dot{=} \mathbb{E}[ R_t | A_t = a ] \qquad \forall a \in \{1, \dots, k\} \\
       &= \sum_r \underbrace{p(r | a)}_{\text{prob. of observing reward} \; r}r
\end{align*}
$$ 

- The __value__ of an action $a$ is the __expected reward__ when that action is taken
- The goal is to __maximize__ the __expected reward__: $argmax_a \, q_*(a)$
- $q_*(a)$ is not known, so we __estimate__ it

<u>Sample-Average Method</u>

$$
\begin{align*}
    Q_t(a) \dot{=} \frac{\text{sum of rewards when} \; a \; \text{taken prior to} \; t}{\text{number of times} \; a \; \text{taken prior to} \; t} = \frac{\sum_{i=1}^{t-1}R_i}{t-1} = \frac{1}{t-1} \sum_{i=1}^{t-1}R_i
\end{align*}
$$

- Can be used to estimate action values $Q(a)$

#### 1.1.2.2. Estimating Action Values Incrementally

- Derived incremental sample average method
- Generalized the __incremental update rule__ into a more __general update rule__
- A __constant stepsize parameter__ can be sued to solve __a non-stationary bandit problem__

<u>Incremental Update Rule</u>

$$
\begin{align*}           
    \underbrace{Q_{t+1}}_{NewEstimate} 
            &= \underbrace{Q_t}_{OldEstimate} +
               \underbrace{\alpha_t}_{StepSize}(
               \underbrace{R_t}_{Target} -
               \underbrace{Q_t}_{OldEstimate})
\end{align*}
$$

### 1.1.3. Exploration vs Exploitation Tradeoff
#### 1.1.3.1. What is the trade-off?

- We discussed the __tradeoff__ between __exploration and exploitation__
- We introduced __epsilon-greedy__ which is a simple method for balancing exploration and exploitation

<u>Exploration versus Exploitation</u>

- __Exploration__ - _improve_ knowledge for _long-term_ benefit
- __Exploitation__ - _exploit_ knowledge for _short-term_ benefit

<u>Epsilon-Greedy Action Selection</u> 

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

- is a method to choose when to exploit and when to explore

#### 1.1.3.2. Optimistic Initial Values

- __Optimistic initial values__ encourage _early exploration_ 
- Described limitations of __optimistic initial values__

<u>Limitations of optimistic initial values</u>

- Optimistic initial values only _drive early exploration_
- They are not well-suited for _non-stationary problems_ 
- We may not know what the _optimistic initial value_ should be

#### 1.1.3.3. Upper-Confidence Bound (UCB) Action Selection

- __Upper-Confidence Bound action-selection__ uses _uncertainty_ in the value estimates for balancing exploration and exploitation

<u>Upper-Confidence Bound (UCB) Action Selection</u>

$$
\begin{align*}
    A_t \dot{=} argmax \Big[ \underbrace{Q_t(a)}_{\text{Exploit}} + c \underbrace{\sqrt{\frac{ln \, t}{N_t(a)}}}_{\text{Explore}} \Big]
\end{align*}
$$

## 1.2. Week 🕑

![problemOfRl]({{ site.baseurl }}/assets/images/problem_of_rl_via_mdp.png)

### 1.2.1. Introduction to Markov Decision Processes

- _MDPs_ provide a general framework for sequential decision making
- The __dynamics__ of an MDP are defined by a probability distribution

<u>Markov property</u>

For a stochastic process, given a sequence of states $S_1, \dots, S_n$, the probability of transitioning to the next state $S_{n+1}$ depends only on the current state $S_n$ and not on the sequence of states that led to $S_n$. 

$$
\begin{align*}
    \mathbb{P}(S_{n+1} | \underbrace{S_n, S_{n-1}, \dots, S_1}_{\text{Entire history up to} \, S_n}) = \mathbb{P}(S_{n+1} | S_n)
\end{align*}
$$

where $\mathbb{P}(S_{n+1} | S_n))$ is the probability of moving to state $S_{n+1}$ given the current state $S_n$. 

- States that the future state of a process depends only on the current state and not on the sequence of events that preceded it
- Posits that the _future is independent of the past_, given the present
- Does not mean that the state representation tells all that would be useful to know, only that it has not forgotten anything that would be useful to know

### 1.2.2. The Goal of Reinforcement Learning

- The __goal of an agent__ is to __maximize the expected return__
- In __episodic tasks__ the agent environment interaction breaks up into __episodes__

<u>Goal of an Agent: Format definition</u>

$$
\begin{align*}
    \mathbb{E}[\underbrace{G_t}_{\text{Rdm. var.}}] = \mathbb{E}[ R_{t+1} + R_{t+2} + R_{t+3} + \dots + \underbrace{R_T}_{\text{Reward of final timestep}} ]
\end{align*}
$$ 

### 1.2.3. Michael Littman: The Reward Hypothesis

<u>Goals as Rewards</u>

- $1$ for goal, $0$ otherwise: goal-reward representation
- $-1$ for not goal, $0$ once goal reached: action-penalty representation

<u>Whence Rewards?</u>

- Programming
  - Coding
  - Human-in-the-loop (source of reward is a human)
- Examples
  - Mimic reward
  - Inverse RL (goes from behavior to rewards. a trainer demostrates an example of the desired behavior and the learner figures out what rewards the trainer must have been maximizing that makes this behavior optimal)
- Optimization (rewards can be derived indirectly through an optimization process. If there's some high-level behavior we can create a score for, an optimization approach can search for rewards that encourage that behavior)
  - Evolutionary optimization 
  - Meta RL (multiple agents)

<u>Challenges to the Reward Hypothesis</u>

- Target is something other than expected cumulative reward:
  - How represent risk-sensitive behavior? 
  - How capture diversity in behavior? 
- Good match for high-level human behavior? 
  - Blind reward pursuers aren't good people
  - We create our "purpose" over years, lifetimes

### 1.2.4. Continuing Tasks

- In __continuing tasks__, the agent-environment interaction goes on indefinitely
- __Discounting__ is used to ensure returns are finite
- Return can be defined __recursively__

| Episodic Tasks | Continuing Tasks | 
| -------------- | ---------------- | 
| Interaction breaks naturally into __episodes__ | Interaction goes on __continually__ | 
| Each episode ends in a __terminal state__ $T$ | No __terminal state__ | 
| Episodes are __independent__ | |
| $G_t \dot{=} R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T$  | $G_t \dot{=} R_{t+1} + R_{t+2} + R_{t+3} + \cdots = \infty?$ |  

<u>Recursive nature of returns</u>

$$
\begin{align*}
    G_t &\dot{=} R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \\
        &= R_{t+1} + \gamma ( \underbrace{R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \cdots}_{G_{t+1}} ) \\
    G_t &= R_{t+1} + \gamma G_{t+1}
\end{align*}
$$ 

<u>Effect of $\gamma$ on agent behavior</u>

$$
\begin{align*}
    G_t &\dot{=} R_{t+1} + \color{Violet}{\gamma} R_{t+2} + \color{Violet}{\gamma^2} R_{t+3} + \cdots + \color{Violet}{\gamma^{k-1}} R_{t+k} + \cdots \\
        &= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\end{align*}
$$ 

- $\gamma \in [0, 1)$ is a discounting factor
- Used to __discount__ rewards in the future
- Makes sure that $G_t$ is finite
- Idea is that immediate rewards contribute more to the sum
- If $\gamma = 0$: (_Short-sighted agent_) Agent only cared about the immediate reward. $G_t = R_{t+1}$
- If $\gamma = 1$: (_Far-sighted agent_) Agent takes future rewards into account more strongly. $G_t = R_{t+1} + \gamma G_{t+1}$

Remember that the sum of an infinite geometric series is defined as: 

$$
\begin{align*}
    a + ar + ar^2 + ar^3 + \cdots \\
    S = \sum_{n=1}^{\infty} ar^{n-1} = \frac{a}{1-r}
\end{align*}
$$ 

where $a$ is the start term and $r$ is the common ratio. The common ratio is a value for which the values in a series gets consistently multiplied by [[Wikipedia, Infinite Geometric Series](https://www.idealminischool.ca/idealpedia/index.php/Infinite_Geometric_Series#:~:text=The%20general%20formula%20for%20finding,r%20is%20the%20common%20ratio.)].

## 1.3. Week 🕒

## 1.4. Week 🕓