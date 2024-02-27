---
layout: post
comments: false
title: "Reinforcement Learning Specialization"
date: 2024-02-04 12:00:00
tags: reinforcement-learning
toc: true
---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

This page contains summaries and personal notes obtained during my studying for the [Reinforcement Learning Specializations](https://www.coursera.org/specializations/reinforcement-learning) from the [University of Alberta](https://www.ualberta.ca/index.html) on [Coursera](https://www.coursera.org/). Note that some of the content is completely taken from the course and therefore the credit goes to the University of Alberta and Coursera. 

# 1. Course 1️⃣ - Fundamentals of Reinforcement Learning

## 1.1. Week 🕐 - Introduction

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

## 1.2. Week 🕑 - Markov Decision Processes
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

where $\mathbb{P}(S_{n+1} \vert S_n)$ is the probability of moving to state $S_{n+1}$ given the current state $S_n$. 

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


## 1.3. Week 🕒 - Value Functions & Bellmann Equations


<u>Summary</u>

- __Policies__ tell an agent how to behave in their environment
  - _Deterministic_ policies: Map a state to an action
    - $s \xrightarrow{\pi} a$
    - Choose an action with $\pi(a)$
  - _Stochastic_ policies: Map a state to a distribution of actions over all possible actions
    - $s \xrightarrow{\pi} \Delta (a)$
    - Choose an action with $\pi(a \vert s)$
  - A policy depends only on the __current state__. Not on e.g. time or previous states. This is a restriction on the state, not the agent. 
    - Thus, the state should provide the agent with all the information it needs to make a good decision.
- __Value functions__ estimate future return (= total reward) under a specific policy.
  - Simplify things by aggregating many possible future returns into a single number
  - _State-value_ functions: 
    - $$
    \begin{align*}
    v_{\color{Blue}{\pi}}(\color{Red}{s}) \dot{=} \mathbb{E}_{\color{Blue}{\pi}}[ \color{Green}{G_t} | \color{Red}{S_t = s} ]
    \end{align*}
    $$
    - Expected return from current state $s$, if the agent follows $\pi$ afterwards.
  - _Action-value_ functions: 
    - $$
    \begin{align*}
    q_{\color{Blue}{\pi}}(\color{Red}{s}, \color{Red}{a}) \dot{=} \mathbb{E}_{\color{Blue}{\pi}}[ \color{Green}{G_t} | \color{Red}{S_t = s}, \color{Red}{A_t = a} ]
    \end{align*}
    $$
    - Expected return from state $s$ if the agent first selects $a$ and then follows $\pi$ afterwards
- __Bellmann equations__ define a relationship between the value of a state, or state-action pair, and its possible successor states
  - Bellmann equation for the _state-value_ function: 
    - $$
    \begin{align*}
    v_{\pi}(s) = \sum_{a}\pi(s \vert a) \sum_{s'}\sum_{r} p(s',r | s,a) [r + \gamma v_{\pi}(s')]
    \end{align*}
    $$ 
    - gives the value of the current state as a sum over the values of all the successor states, and immediate rewards
  - Bellmann equation for the _state-action_ function:
    - $$
    \begin{align*}
    q_{\pi}(s, a) = \sum_{s'}\sum_{r} p(s',r | s,a) [r + \gamma \sum_{a'} \pi(a' \vert s') q_{\pi}(s', a')]
    \end{align*}
    $$
    - gives the value of a particular state-action pair as the sum over the values of all possible next state-action pairs and rewards
  - Can be directly used to find the value function
  - Help us evaluate policies
  - But, they can't find a policy that attains as much reward as possible
- __The ultimate goal__: to find a policy that obtains as much reward as possible
  - __Optimal policy__: achieves the highest value possible in every state
    - There is always $\geq 1$ optimal policies
    - _Optimal state-value function_ $v_*$: highest possible value in every state
      - $v_{\pi_*}(s) = \max_{\pi} v_{\pi}(s)$
      - every optimal policy shares the same optimal state-value function
    - _Optimal action-value function_ $q_*$:
      - $q_{\pi_*}(s,a) = max_{\pi} q_{\pi}(s,a) \; \text{for all} \; s \in \mathcal{S} \; \text{and} \; a \in \mathcal{A}$ 
    - __Optimal Bellmann equations__
      - $$
      \begin{align*}
      v_{\pi_*}(s) = max_a \sum_{s'}\sum_{r} p(s',r | s,a) [r + \gamma v_*(s')]
      \end{align*}
      $$ 
      - $$
      \begin{align*}
      q_{\pi_*}(s, a) = \sum_{s'}\sum_{r} p(s',r | s,a) [r + \gamma \, max_a q_*(s', a')]
      \end{align*}
      $$

### 1.3.1. Bellmann Equation Derivation

- We have derived the __Bellmann Equations__ for __state-value__ and __action-value functions__
- The current time-step's __state/action values__ can be written recursively in terms of __future state/action values__

<u>State-value Bellmann equation</u>

$$
\begin{align*}
    v_{\pi}(s) &\dot{=} \color{Green}{\mathbb{E}_{\pi}[ G_t | S_t = s ]} \\
               &= \color{Red}{\mathbb{E}_{\pi} [} \color{Blue}{R_{t+1} + \gamma G_{t+1}} \color{Red}{| S_t = s]} 
                \qquad \qquad \text{recalling that} \; \color{Blue}{G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}}
               \\
               &= \color{Red}{\sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a)} \Big[ \color{Blue}{r + \gamma} \color{Green}{\mathbb{E}_{\pi} [ G_{t+1} | S_{t+1} = s']} \Big] \\
               &= \color{Red}{\sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a)} \Big[ \color{Blue}{r + \gamma} \color{Green}{v_{\pi}(s')} \Big] \\
\end{align*}
$$ 

<u>Action-value Bellmann equations</u>

$$
\begin{align*}
    q_{\pi}(s, a) &\dot{=} \color{Green}{\mathbb{E}_{\pi}[ G_t | S_t = s, A_t = a ]}
    \\
                  &= \color{Red}{\mathbb{E}_{\pi}[} G_t \color{Red}{| S_t = s, A_t = a ]}
    \\
                  &= \color{Red}{\sum_{s'}\sum_{r} p(s', r | s, a)} \Big[ r + \gamma \mathbb{E}_{\pi}[ G_{t+1} | S_{t+1} = s']  \Big]
    \\
                  &= \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma \sum_{a'}\pi (a' | s') \color{Green}{\mathbb{E}_{\pi}[G_{t+1} | S_{t+1}=s', A_{t+1} = a']} \Big]
    \\ 
                  &= \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma \sum_{a'}\pi (a' | s') \color{Green}{q_{\pi}(s', a')} \Big]
\end{align*}
$$ 

### 1.3.2. Why Bellmann Equations? 

- You can use the __Bellmann Equations__ to solve for a __value function__ by writing a __system of linear equations__. (e.g. a linear equation for each action in each state)
- Without the Bellman equation, we might have to consider an infinite number of possible futures
- We can only solve __small MDPs__ directly, but __Bellmann Equations__ will factor into the solutions we see later for __large MDPs__

## 1.4. Week 🕓 - Dynamic Programming

<u>Summary</u>

- __Policy evaluation__ is the task of determining the state-value function $v_{\pi'}$ for a particular policy $\pi$. 
  
$$
\begin{align*}
    v_{\color{Green}{\pi}}(s) &{\color{Green}{=}} \sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma v_{\color{Green}{\pi}}(s') \Big] \\
    v_{\color{Red}{k+1}}(s) &{\color{Red}{\leftarrow}} \sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma v_{\color{Red}{k}}(s') \Big] \\
\end{align*}
$$ 

- __Control__ refers to the task of improving a policy

<u>Policy Improvement Theorem</u>

$$
\begin{align*}

\pi'(s) \dot{=} argmax_a \sum_{s'}\sum_{r} p(s',r \vert s,a) [ r + \gamma v_{\pi}(s') ] \\
\pi' > \pi \; \text{unless} \; \pi \; \text{is optimal}

\end{align*}
$$

- Generalized Policy Iteration includes __asynchronous__ DP methods


### 1.4.1. Policy Evaluation (Prediction)

#### 1.4.1.1. Policy Evaluation vs Control

- __Policy evaluation__ is the task of determining state-value function $v_{\pi}$, for a particular policy $\pi$
- __Control__ is the task of improving an existing policy 
- __Dynamic programming__ techniques can be used to solve both these tasks, if we have access to the __dynamics function__ $p$

#### 1.4.1.2. Iterative Policy Evaluation

- We can turn the Bellmann equation into an __update rule__, to __iteratively__ compute value functions 

<u>Iterative Policy Evaluation in a Nutshell</u>

$$
\begin{align*}
    v_{\color{Green}{\pi}}(s) &{\color{Green}{=}} \sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma v_{\color{Green}{\pi}}(s') \Big] \\
    v_{\color{Red}{k+1}}(s) &{\color{Red}{\leftarrow}} \sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma v_{\color{Red}{k}}(s') \Big] \\
\end{align*}
$$ 

This gives us a procedure to iteratively refine our estimate of the value function. Each iteration produces a better estimate. If $v_{k+1} = v_k$ for all states $s$ then $v_k = v_{\pi}$ and thus we have found the value function. This is because $v_{\pi}$ is the unique solution to the Bellmann equation. In fact, 

$$
\begin{align*}
    \text{For any} \; v_0: \; \lim_{k \to \infty} v_k = v_{\pi}
\end{align*}
$$

- _Sweep_: Once each iteration applied the update to every state $s \in \mathcal{S}$.

We simply need to store two arrays, $V$ (for the current value estimates) and $V'$ (for the updated value estimates). We update $V'$ one state at the time, where every estimate in $V$ integrates into the update of the current state in $V'$. At the end of a full sweep we can set $V \gets V'$. Then we do the next iteration.

$$
\begin{align*}

& \underline{\phantom{\textbf{Iterative Policy Evaluation, for estimating} \; V \approx v_{\pi}}} \\
& \underline{\textbf{Iterative Policy Evaluation, for estimating} \; V \approx v_{\pi}} \\
& \text{Input:} \; \pi && \color{Grey}{\text{the policy to be evaluated}} \\
& V \gets \vec{0}, V' \gets \vec{0} \\ 
& \text{Loop:} \\  
& \qquad \Delta \leftarrow 0 \\ 
& \qquad \text{Loop for each} \; s \in S: && \color{Grey}{\text{a sweep}} \\ 
& \qquad \qquad V'(s) \leftarrow \sum_{a}\pi(a \vert s) \sum_{s', r} p(s',r|s,a) [ r + \gamma V(s') ] \\ 
& \qquad \qquad \Delta \leftarrow \max(\Delta , | V'(s) - V(s)) && \color{Grey}{\text{largest update to this state value}} \\ 
& \qquad V \leftarrow V' \\ 
& \text{until} \; \Delta < \theta \quad \text{(a small positive number)} && \color{Grey}{\text{value function stops changing}} \\
& \text{Output:} \; V \approx v_{\pi}

\end{align*}
$$

### 1.4.2. Policy Iteration (Control)

#### 1.4.2.1. Policy Improvement

<details>
<summary>Content of Section:</summary>

<ul>
<li> <i>Understand</i> the <b>policy improvement theorem</b> </li>
<li> <i>Use</i> a value function for a policy to produce a better policy for a given MDP </li>
</ul>

</details>

- The __Policy Improvement Theorem__ tells us that a greedified policy is a strict improvement (unless the original policy was already optimal)
- Use the value function under a given policy, to produce a strictly better policy

Let's recall that 

$$
\begin{align*}

\pi_*(s) &= \underbrace{argmax_{a}}_{\text{greedy action}} \sum_{s'}\sum_{r}p(s',r | s,a) [ r + \gamma v_*(s') ]
\\ 
\pi_{s} &= argmax_{a} \sum_{s'}\sum_{r}p(s',r | s,a) [ r + \gamma v_{\pi}(s') ] \; \text{for all} \; s \in \mathcal{S}
\\
&\rightarrow \; v_{\pi} \; \text{obeys the Bellmann optimality equation} 
\\ 
&\rightarrow \; \pi \; \text{is optimal}

\end{align*}
$$

<u>Policy Improvement Theorem</u>

> The new policy is a strict improvement over $\color{Blue}{\pi}$ unless $\color{Blue}{\pi}$ is already optimal

$$
\begin{align*}

q_{\color{Blue}{\pi}}(s, \color{Red}{\pi'}(s)) \geq q_{\color{Blue}{\pi}}(s, \color{Blue}{\pi}(s)) \; \text{for all} \; s \in \mathcal{S} \; \rightarrow \color{Red}{\pi'} \geq \color{Blue}{\pi} 
\\
q_{\color{Blue}{\pi}}(s, \color{Red}{\pi'}(s)) > q_{\color{Blue}{\pi}}(s, \color{Blue}{\pi}(s)) \; \text{for at least one} \; s \in \mathcal{S} \; \rightarrow \color{Red}{\pi'} > \color{Blue}{\pi} 

\end{align*}
$$

Thus, the value function of a given policy can be used to find a better policy. Now, how can we use this to find the optimal policy by iteratively evaluating and proving a sequence of policies 👇

#### 1.4.2.2. Policy Iteration

<details>
<summary>Content of Section:</summary>

<li> <i>Outline</i> the <b>policy iteration</b> algorithm for finding the optimal policy </li>
<li> <i>Understand</i> the <b>dance of policy and value</b>, how policy iteration reaches the optimal policy by alternating between evaluating a policy and improving it</li>
<li><i>Apply</i> policy iteration to compute optimal policies and optimal value functions</li>

</details>

- __Policy Iteration__ works by alternating __policy evaluation__ and __policy improvement__
- Policy Iteration follows a sequence of __better and better policies and value functions__ until it reaches the optimal policy and associated optimal value function

The intuition of policy iteration is to evaluate the current value, yielding in the value function for the current policy. We then use this value function to obtain a new policy. We continue this evaluation ($\color{Green}{E}$) and improvement ($\color{Blue}{I}$) play nutil we found the optimal value function and therefore the optimal policy.

$$
\begin{align*}

\pi_1 \xrightarrow{\color{Green}{E}} v_{\pi_1} \xrightarrow{\color{Blue}{I}} \pi_2 \xrightarrow{\color{Green}{E}} v_{\pi_2} \xrightarrow{\color{Blue}{I}} \pi_3 \xrightarrow{\color{Green}{E}} \cdots \xrightarrow{\color{Blue}{I}} \pi_* \xrightarrow{\color{Green}{E}} v_{\pi_*} \xrightarrow{\color{Blue}{I}} \pi_*

\end{align*}
$$

Each policy in this cycle is _deterministic_ and since there are finite number of deterministic policies, this process must eventually reach an optimal policy. Note that always

$$
\begin{align*}

\pi_{i+1} \; \text{is greedy wrt} \; v_{\pi_i}

\end{align*}
$$

but $v_{\pi_i}$ no longer accurately reflects the value of $\pi_{i+1}$. Only the next $\color{Green}{E}$ step makes the value function $v_{\pi_{i+1}}$ accurately reflect the value of $\pi_{i+1}$. 

![policyIteration]({{ site.baseurl }}/assets/images/rl_specialization/policy_iteration.webp)

The following shows the policy iteration algorithm in pseudocode.

$$
\begin{align*}

& \underline{\phantom{\textbf{Policy Iteration (using iterative policy evaluation) for estimating} \; \pi \approx \pi_*}} \\
& \underline{\textbf{Policy Iteration (using iterative policy evaluation) for estimating} \; \pi \approx \pi_*} \\
& 1. \; \text{Initialization} \\
& \phantom{1.} \; V(s) \in \mathbb{R} \; \text{and} \; \pi(s) \in \mathcal{A}(s) \; \text{arbitrarily for all} \; s \in \mathcal{S} \\  
\\
& 2. \; \text{Policy Evaluation} \\
& \phantom{2.} \; \text{Loop:} \\ 
& \qquad \Delta \leftarrow 0 \\ 
& \qquad \text{Loop for each} \; s \in S: \\ 
& \qquad \qquad V(s) \leftarrow \sum_{s',r}\pi(s',r \vert s,\pi(s)) [ r + \gamma V(s') ] \\ 
& \qquad \qquad \Delta \leftarrow \max(\Delta , | v - V(s)) \\
& \phantom{2.}  \; \text{until} \; \Delta < \theta \quad \text{(a small positive number determining the accuracy of estimation)} \\
\\
& 3. \; \text{Policy Improvement} \\
& \phantom{2.} \; policyStable \gets true \\
& \qquad \text{For each} \; s \in \mathcal{S}: \\  
& \qquad \qquad oldAction \gets \pi(s) \\
& \qquad \qquad \pi(s) \gets argmax_a \sum_{s',r} p(s',r \vert s,a) [ r + \gamma V(s') ] \\
& \qquad \qquad \text{If} \; oldAction \neq \pi(s), \; \text{then}\; policyStable \gets false \\
& \qquad \text{If} \; policyStable, \; \text{then stop and return} \; V \approx v_*, \text{and} \; \pi \approx \pi_*; \text{else go to} \; 2

\end{align*}
$$

### 1.4.3. Generalized Policy Iteration

#### 1.4.3.1. Flexibility of the Policy Iteration Framework

<details>
<summary>Content of Section:</summary>

<li> <i>Understand</i> the framework of <b>Generalized Policy Iteration</b></li>
<li> <i>Outline</i> <b>Value Iteration</b>, an important special case of Generalized Policy Iteration</li>
<li><i>Apply</i> policy iteration to compute optimal policies and optimal value functions</li>
<li> <i>Understand</i> the distinction between <b>synchronous</b> and <b>asynchronous</b> dynamic programming methods </li>

</details>

- __Value Iteration__ allows us to combine policy evaluation and improvement into a single update
- __Asynchronous__ dynamic programming (DP) methods give us the freedom to update states in any order
- __Generalized Policy Iteration__ unifies classical DP methods, value iteration, and asynchronous DP

Value iteration is a generalized policy iteration algorithm. In value iteration, we sweep over all the states and greedify wrt the current value function. However, we do not run $\color{Green}{E}$ to completion. We just perform one sweep over all the states. After that, we greedify again. The algorithm looks very similar to iterative policy evaluation, however, instead of updating the value according to a fixed policy, we update using the action that maximizes the current value estimate.

$$
\begin{align*}

& \underline{\phantom{\textbf{Value Iteration for estimating} \; \pi \approx \pi_*}} \\
& \underline{\textbf{Value Iteration for estimating} \; \pi \approx \pi_*} \\
& \text{Algorithm parameter: a small threshold} \; \theta > 0 \; \text{determining accuracy of estimation} \\
& \text{Initialize} \; V(s), \text{for all} \; s \in \mathcal{S^{+}}, \text{arbitrarily except that} \; V(terminal)=0 \\
\\
& \text{Loop:} \\
& \qquad \Delta \gets 0 \\
& \qquad \text{Loop for each} \; s \in S: \\ 
& \qquad \qquad v \gets V(s)  \\
& \qquad \qquad V(s) \gets max_a \sum_{s',r}\pi(s',r \vert s,\pi(s)) [ r + \gamma V(s') ] \\
& \qquad \qquad \Delta \leftarrow \max(\Delta , | v - V(s)) \\
& \text{until} \; \Delta < \theta \\
\\
& \text{Output a deterministic policy,} \; \pi \approx \pi_*, \; \text{such that} \\
& \qquad \pi(s) = argmax_a \sum_{s',r} p(s',r \vert s,a) [ r + \gamma V(s') ]

\end{align*}
$$

Value iteration sweeps the entire state space in each iteration, called a full sweep. Thus, value iteration and policy iteration are called _synchronous_. Obviously, this is a problem for large state spaces. 

_Asynchronous_ DP algorithms update the values of states in any order, they do not perform systematic sweeps. In order to guarantee convergence, asynchronous algorithms must continue to update the values of all states.

#### 1.4.3.2. Efficiency of Dynamic Programming

<details>
<summary>Content of Section:</summary>

<li> <i>Describe</i> <b>Monte Carlo Sampling</b> as an alternative method for learning a value function</li>
<li> <i>Describe</i> <b>Brute-Force Search</b>as an alternative for finding an optimal policy</li>
<li> <i>Understand</i> the advantage of <b>DP</b> and <b>bootstrapping</b> over these alternative strategies for finding the optimal policy </li>

</details>

- __Bootstrapping__ can save us from performing a huge amount of unnecessary work

<u>Bootstrapping</u>

> Using the value estimates of successor states $v(s')$ to improve the current value estimate $v(s)$

$$
\begin{align*}
    v_{k+1}(\color{Red}{s}) &= \sum_{a}\pi (a | s) \sum_{s'}\sum_{r} p(s', r | s, a) \Big[ r + \gamma v_{k}(\color{Red}{s'}) \Big]
\end{align*}
$$ 

<u>Efficiency of DP</u>

Policy iteration is guaranteed to find the optimal policy in time polynomial in the number of states $\vert \mathcal{S} \vert$ and actions $\vert \mathcal{A} \vert$. 

<u>Curse of Dimensionality</u>

> The size of the state space grows __exponentially__ as the number of relevant features increases.


## 1.5. References

[1] Richard S. Sutton and Andrew G. Barto. ["Reinforcement Learning: An Introduction."](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) 2nd edition (2014), MIT Press.
