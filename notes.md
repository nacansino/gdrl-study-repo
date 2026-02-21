# Notes

## Creating environment

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

### Setup

Run the setup script:
```bash
./environment.sh
```

### Running Commands

With uv, you don't need to activate the virtual environment. Use `uv run`:
```bash
# Start Jupyter Lab
uv run jupyter lab

# Run a Python script
uv run python my_script.py
```

## Changing from `gym` to `gymnasium`

### API change 1: return value of env.reset()

- Old: `state = env.reset()`
- New: `state, info = env.reset()`

### API change 2: return value of env.step(action)

- Old: `state, reward, done, info = env.step(action)`
- New: `state, reward, terminated, truncated, info = env.step(action)`

### Updating code examples snippets in the book should be updated to reflect these changes. For example:

```python
# Old code
state = env.reset()
...
state, reward, done, info = env.step(action)

# New code
state, info = env.reset()
...
state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```
## Custom Environments

When creating custom environments, ensure that the `reset` and `step` methods conform to the new API specifications of `gymnasium`.

### Reset Method

```python
def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)    # very important to call the super method
    ...
    return observation, info
```

### Step Method

```python
def step(self, action):
    ...
    return observation, reward, terminated, truncated, info
```

### Using `self.np_random`
Before using `self.np_random`, ensure to initialize it in the `reset` method:

For example, when subclassing a custonm environment `CustomEnv`:


```python
class CustomEnv2(CustomEnv):
    # At this point, self.np_random is not initialized
    # but I want to use random function when initializing action space
    def __init__(self):
        p_dist = self.np_random.integers(2, 5)  # This will work but will not be reproducible
        super().__init__()
```

```python
class CustomEnv2(CustomEnv):
    def __init__(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)    # Call reset to initialize self.np_random
        p_dist = self.np_random.integers(2, 5)  # Now this will work and be reproducible
        super().__init__()
```

## Speed Notes RL

### Chapter 2

We started Chapter 2 which is about modeling environments using MDP (Markov Decision Processes). It is composed of states, actions, and rewards. MDP is defined by a transition matrix (or transition probability). 
MDP is not always available, but in toy problems usually they are baked inside the `environment` which is provided by `Gymnasium` or any other simulator you have.
MDP can either be deterministic (doing a certain action in a state always guarantees the same next state every time you do it), or stochastic (doing the same action in a state may land you in different next states).

The API for an MDP is 
- Input: state, action
- Output: next_state, reward

**Terms**:
- `MDP`: mathematical framework to model any complex sequential decision-making problem under uncertainty
- `agent` is the decision maker and is providing the solution to a problem
- `environment` is the representation of a problem; anything that is not the decision maker is part of the environment. It is represented by a set of variables related to the problem.
- `observation` are set of variables that the agent perceives at any given time.
- `action` is what the agent does in the environment; mechanism to influence the environment
- `action space` is the set of all possible action in all states. A state may allow only a subset of the action space.
- `transition function` refers to how (state, action) pair yields the (next_state). This can be deterministic or stochastic, in the latter case it is defined as a probability distribution.
- `Reward` is what you get for entering a certain state.
- `Reward function` refers to how (state, action) pair yields the reward
- `model of the environment` refers to transition function and reward function. If you have the two, then you have the model of the environment.
- `absorbing` or `terminal state` is a state where all available actions transitioning deterministically to itself, and these transitions provide no reward
- `time step` is a global clock syncing all parties and discretizing time.
- `episode` is a task where there's a finite number of timesteps, because the agent reaches a terminal state or the clock stops
- `discount` is a value we associate to future rewards. "the further into the future we receive the reward, the less valuable it is in the present". It is represented by gamma.
- `Returns` is the sum of all rewards obtained during the course of an episode (normally discounted with gamma). It answers the question "how much you got?" at the end of an ep.

### Chapter 3

Here we learned about algorithms that solve MDP. We learned Policies, `State Value function`, `Action-Value function`, `Action-Advantage function`.
We also discussed `optimality`, which means if the terms above are the best they can be. The goal in RL is to find the `optimal policy`, and often times as a result we also get the `Optimal action-value function`.

Given an MDP, we can do `planning` in order to solve for the optimal policy. We discussed two planning algorithms, `Policy Iteration` or `Value Iteration`.

We first learn how to quantize how good a policy is by using `policy evaluation`. Once we know how good a policy is through V_pi, we now improve it by getting the Action-Value function Q_pi from V_pi and the MDP, which allows us to get an improved policy. This is called `policy improvement`.

If you do `policy evaluation` and `policy improvement` repeatedly until the policy does not change anymore, you get the optimal policy. You call this process `policy iteration (PI)`.

An improvement to this process is when during the `policy evaluation` stage, V is calculated only in one sweep (no long loop where we wait for V to stabilize), and use this to improve the policy (calculate Q, and improved pi). We then use this pi value to do policy evaluation again, until V stabilizes. This improved version is called `Value iteration (VI)` and is shown to converge to the optimal policy similar to policy iteration.

Both of them are instances of `generalized policy iteration (GPI)`. 

**Terms**
- `Planning` refers to the use of a model of the environment in order to find a policy
- `Policy (pi)` is a mapping between state->action. Answers the question "when in a certain state, what do you do?"
- `State-Value function (V)` is the *expected* returns in all states when following a certain policy. Mathematically it is `V_pi[s]`.
- `Action-Value function (Q)` is the *expected* returns in all states given a certain action is selected on a state, and following a certain policy pi thereafter `Q_pi[s][a]`.
- `Action-Advantage function (A)` is the difference between the action value function and the state value function. `A_pi[s][a] = Q_pi[s][a] - V_pi[s]`. It describes how much better it is to take action `a` compared to just following `pi` at state `s`.
- `Optimal policy` - is a policy where the expected returns V are greater than or equal to any other policy, for ALL states
- `Optimal action-value function` - an action-value function with the maximum value across all policies for all state-action pairs.
- `Policy evaluation`: get V of a certain policy (run bellman equation). It can be done using dynamic programming. 
- `Policy improvement`: getting a better policy by computing Q from V and the MDP.
- `Policy iteration`: is doing policy evaluation and policy improvement until policy stabilizes
- `Value iteration`: is a modification of policy iteration where we combine policy evaluation and policy improvement in a single update step.
- `generalized policy iteration (GPI)`: general idea in RL in which policies are improved using their value function estimates, and value function estimates are improved toward the actual value function for the current policy

### Chapter 4

We learned from previous chapter how to solve RL environments using planning algorithms if we have access to the MDP. However, we can not always assume we know with precision how an environment reacts to our actions. So in this chapter we explore techniques how to solve RL environments without having access to the MDP, through trial-and-error learning. Tradeoff between exploration and exploitation is examined here.

In order to explain the concept, the Multi-Armed Bandits (MAB) environment was introduced. The special feature of that is that the size of the state space and horizon is 1. Meaning there is no *sequence*, doing a single action immediately terminates the program. So we don't need to consider credit assignments to each state, we only consider techniques how to learn Q. This environment is **nonsequential** but **evaluative**.

**Approaches**:
- `random exploration strategy`: select greedy action most of the time, but by chance chooses random action (epsilon)
- `optimistic exploration strategies`: quantifies the uncertainty in the decision-making problems & increases the preference for states with the highest uncertainty
- `information state-space exploration strategies`: we add another dimension that describes the uncertainty of the Value of the state (i.e.) `s' = (s, information)`

**Specific algorithms covered**:
- `greedy` or `pure_exploitation`: always pick the action based on the highest estimated action-value
- `random` or `pure_exploration`: always pick a random action
- `epsilon-greedy`: do greedy sometimes, do random otherwise
- `decaying epsilon-greedy`: explore early on, then eventually exploit
- `optimistic initialization`: initialize Q to a high value then act greedily towards it
- `softmax`: pick an action using the Q-values as probability. Essentially using higher Q state-action as preferred action.
- `Upper confidence bound (UCB)`: we augment Q during argmax selection by adding an uncertainty bonus, higher when least explored, lower when explored enough already.
- `thompson sampling`: uses Bayesian technique to obtain action relative to Q

### Chapter 5

In the previous chapter we emphasized on learning when we don't have access to the MDP but simplifying it to nonsequential environments. 
This chapter talks about sequential environments where we focus on techniques on learning the `State-Value function (V)` of a policy in **sequential** and **evaluative** MDPs, so now we have more than one state and an episode can run with more than one timestep. This problem is often called the `prediction problem` because our goal is to estimate the value function.

Note we are still not learning the optimal policy here, in this chapter we are given a policy and our task is to learn the Value function. As an analogy, these methods are like `Policy Evaluation` discussed in Chapter 3, except that we don't need MDP in order to get the value of a policy.

Below are the algorithms covered.

**Specific algorithms covered**:
- `First visit Monte Carlo (FVMC)`: using policy pi, generate trajectory (until end of episode). Then step through the trajectory. We use the returns G[t:T] (summation of rewards until termination) to compute the target V. V[s] of states are only updated once during first visit.
- `Every-visit Monte Carlo (EVMC)`: FV monte carlo but the value of V is updated on every visit
- `Temporal Difference (TD)`: Compared to MC, instead of using G as target, we use the current estimation V[s] in order to compute the TD target. As an effect we need not to run the trajectory to completion but can update every step
- `n-step TD (NTD)`: A mix of MC and TD, Use the value of G for the first N steps, and use the estimation of V[s] for the succeeding states.
- `TD(lambda)`: Use the concept of **eligibility traces** where we keep track of states where we've been, then update V according to these traces. Mathematically it's like lambda-weighing n-step TD's. If lambda=1 we are weighing the actual returns G compared to the estimated returns V so TD(lambda=1) just becomes normal `MC` algorithm. If lambda=0 we use the estimated value V which is equivalent to normal `TD`.

### Chapter 6

These chapter now focuses on **finding optimal policies** in the RL environments. We work on top of prediction algorithms discussed in Chapter 5 and promote them to yield optimal policies. Thus, we now move on to doing `control` as opposed to only `prediction` in Chapter 5. Now we do both `policy evaluation` and `policy improvement`.

**Specific algorithms covered**:
- `Monte Carlo Control (MC)`: Implements Monte Carlo to predict Q, and uses Epsilon-greedy policy improvement to derive the behavior policy. This is repeatedly done for a given number of episodes.
- `SARSA`: Contrary to MC which finishes an episode before updating the policy, Sarsa uses Temporal Difference (TD) learning to estimate the action-value function Q*. It also uses epsilon-greedy policy improvement to select the new behavior policy. Since we use the same behavior policy and target policy, we call this "on-policy".
- `Q-Learning`: Similar to SARSA, except that in the calculation of TD target, we select the next action greedily and not based on the behavior policy. (this separation makes Q-learning an "off-policy" since behavior and target policy is the same)
- `Double-Q Learning`: Similar to Q-learning, but contrary to Q-learning, Double Q-learning maintains two separate Q functions (Q1 and Q2) to reduce the overestimation bias of Q-learning. In one case, Q1 is used to generate behavior, and the other is used as target. In the other case, Q2 is used to generate behavior and the other is used as target. The algorithm chooses between one or the other in a 50-50 fashion.

**Terms**
- `behavior policy`: the policy used to generate behavior (i.e., to select actions)
- `target policy`: the policy that is being evaluated and improved
- `Off-policy vs. On-policy`: On-policy algorithms use the same policy for both behavior and target policy (e.g., SARSA). Off-policy algorithms use different policies for behavior and target (e.g., Q-learning).

### Chapter 7

These chapter laid out improvements in the algorithms covered in Chapter 6. We still deal with environments that are both **sequential** and **evaluative**.

**Specific algorithms covered**:
- `SARSA(lambda)`: Same as SARSA, but uses eligibility traces (either accumulating trace or replacing trace) to distribute credit assignments to previous states
- `Watkins's Q(lambda)`: Same as Q-learning (uses the greedy policy as target), but uses eligibility traces (either accumulating trace or replacing trace) to distribute credit assignments to previous states
- `Dyna-Q`: A model-based RL technique. It's similar to Q-learning but there is a planning loop where we build the model through `T_count(s,a,s')` and `R_count(s,a,s')` by randomly sampling previously visited state-action pairs, then updating Q using this model. Given enough episodes we see that the model is able to learn the MDP. Key: samples **random one-step transitions from anywhere** in the state space `n` times (a hyperparameter).
- `Trajectory Sampling`: A model-based RL technique. Similar to Dyna-Q, during planning we run full trajectories from time to time, instead of random one-step transitions n times. In doing so we either use on-policy or off-policy to run the trajectory instead of random action.

**Algorithm Comparison (based on Claude)**:

| Algorithm | Sample Efficiency | Robustness to Approximation | Best Use Case |
|-----------|-------------------|-----------|---------------|
| SARSA(λ) | Moderate | High | Safe exploration; risk-sensitive tasks |
| Q(λ) | Moderate-High | Moderate-Low | When optimal policy is priority over safety |
| Dyna-Q | High | Moderate | When environment model is accurate/learnable |
| Trajectory Sampling | High (compute-efficient) | Low-Moderate | Large state spaces; focused exploration |

**Key Research Findings**:

- **Replacing vs. Accumulating Traces**: Replacing traces generally outperform accumulating traces in practice. Accumulating traces can lead to unbounded trace values and instability, especially in tasks with loops or repeated state visits. Replacing traces are more robust.

- **SARSA(λ) vs. Q(λ)**: SARSA(λ) tends to learn safer policies because it accounts for exploration in its updates (on-policy). Q(λ) can be more aggressive but suffers from trace-cutting—when an exploratory action is taken, the eligibility trace is reset, reducing the benefit of λ in highly exploratory settings.

- **λ Selection**: Higher λ (closer to 1) speeds up credit assignment in sparse reward settings but increases variance. Lower λ is more stable but slower to propagate rewards. Empirically, λ ∈ [0.7, 0.9] often works well.

- **Dyna-Q Strengths/Weaknesses**: Dramatically improves sample efficiency when the learned model is accurate. However, in stochastic or complex environments, model errors compound and can degrade performance (model bias problem).

- **Trajectory Sampling vs. Dyna-Q**: Trajectory sampling focuses computational effort on states likely to be visited under the current policy, making it more efficient in large state spaces. Dyna-Q's uniform random sampling wastes effort on irrelevant states. Research shows trajectory sampling converges faster in practice for most MDPs.

- **On-policy vs. Off-policy with Traces**: Eligibility traces work more naturally with on-policy methods. Off-policy methods with traces (like Watkins's Q(λ)) require trace cutting or importance sampling corrections, which can reduce their effectiveness.

### Chapter 8

In the previous chapters, we learned how to solve RL problems in an *exhaustive* manner, meaning we make a table of values for the state-value (V) or action-value (Q) functions that we use to solve for the optimal policy. These approaches we have done so far, for both planning (MC, TD, TD-lambda) and control (MC-Control, SARSA/SARSA-lambda, Q-Learning, Watkins's Q-lambda, Dyna-Q, Trajectory Sampling) are all tabular methods.

But tabular methods are sample-inefficient. Meaning when we are running these algorithms, we only update one value in the Q or V tables, and have to run multiple episodes just to learn. Now, what happens if we have a lot of states like the 8-bit Atari Games (160px x 192px x 255), or infinite-states like Cartpole or in robotics. These table methods become impractical to use.

Now comes non-linear function approximator methods. Non-linear since our targets are usually non-linear. The approach is not to *exhaustively* tabulate state-value V or action-value functions, but to *approximate* these targets. We can use several function approximator but this chapter focuses on *deep reinforcement learning* which hints on using neural networks as function approximators, but in general other non- neural network functions can be used (ex. kernel methods, tree-based model).

**Specific algorithms covered**:
- `NFQ (Neural Fitted Q-iteration)` : vanilla deep RL algorithm that uses neural networks to approximate the action-value function Q. The algorithm is similar to Q-learning, except that instead of using a table to store Q values, we use a neural network. Experiences are collected per episode, and the model updated every batch_size.

There are several knobs to tune here:
1. Value target: Could be v(s), q(s,a), or action-advanage a(s,a)
2. Selecting a non-linear function: here we use a neural network architecture with state-action-in-value-out (s,a -> q(s,a)), or state-in-values-out (s->[q(s|a1) q(s|a2)]). In the book implementation we used state-in-values-out.
3. Loss: Minimizing the loss with respect to the optimal action-value function q*, that is L = difference between q* - Q(s,a;theta). But which q* do we use if we don't have access to it? See #4. Which metric do we use? See #6.
4. Target for policy evaluation: MC, TD, N-step, TD(lambda). Sometimes we also call the on-policy version the `SARSA target` and the off-policy version the `Q-learning` target.
5. Exploration strategy : greedy, epsilon-greedy, softmax, etc.
6. Loss Function: The particular metric to use. MSE (L2), or L1.
7. Optimizaton method: select similar to typical supervised learning problem.

### Chapter 9

In the NFQ introduced in chapter 8, there are two major problems. (1) it violates the independently and identically distributed (IID) assumption of the data, and (2) the stationarity of targets.

IID is violated because the samples are **not** independent since they come from a trajectory. The sample at time t+1 is dependent on the sample at time t. They are also not identically distributed because they depend on the policy that generates the action.

The targets are not stationary since everytime we update the value function, the target changes too since we use the same value function to generate the targets.

In this chapter we addressed these concerns by introducing two mechanisms:

1. Using `Target Networks` : use a separate network that we can fix for multiple steps & reerve it for calculating more stationary targets. We call this the `target network`.
2. Using `Experience Replay` : a buffer of experiences bigger than NFQ, but here we sample mini-batches at random from the buffer of experiences.

**Specific Algorithms Covered**
- `DQN` : NFQ that uses target networks and experience replay
- `Double DQN` : DQN but instead we used double-learning - one for generating the behavior and one used as a target. The target is updated periodically, but the learning only happens with the behavior Q.

### Chapter 10

In this chapter we primarily introduce mechanisms to further improve sample efficiency of the value-based DRL methods introduced before.

Mechanisms:

1. Using a functional NN that splits the Q-function into V(s) and A(s,a). This allows us to squeeze information from samples coming from all actions through V and not just a single Q value for a particular (s,a).
An important thing to remember here is that there is an `aggregating equation` to get Q from V and A, plain addition do not work here.

2. `Prioritized Experience Replay (PER)`: using the idea of prioritizing experiences that are the *most promising* for learning. In RL, this measure of surprise is given by the TD error. Inside the experience tuple we attach the absolute TD error as an element and use that as score to pull out the top experiences. Here we don't just naively rank the experiences by score, but we have a couple of techniques how to use this score:
- Draw experiences stochastically with probability proportional to the score
- Rank-based prioritization (priorities as the reciprocal of the rank of the sample)

Now. Using one distribution for estimating another one introduces bias in the estimates. So we also need to do "weighted importance sampling" by scaling the TD errors by weights calculated with the probabilities of each sample.

3. `Polyak Averaging`: mixing in online network weights into the target network on every step in small percentages

**Specific Algorithms Covered**
- `Dueling DQN` - uses state-value V and advantage function A instead of a single Q.

### Chapter 11

In this chapter, we directly find the optimal policy instead of estimating the value function(s) through methods called `policy-gradient methods`. This solves several issues with value function estimation:

1. State aliasing : in some partially observable MDPs / POMDPs, we might perceive the value or Q of one state as similar or equal to the value of another state, as in Q(s1, a1) ~ Q(s2,a2) when s1 is not equal to s2.
2. Computing the target for hi-dimensional or continuous action spaces. When computing the target policy, we get the max of Q over all a's. This is easy if the action space is discrete, but an O(|A|) problem. In the cartpole environment, the action space is just F=-1 or F=1 which is lo-dimensional discrete. What hapens if we can apply various forces, or what about continuous values? Then computing max(Q(s,a)) now becomes burdensome.
3. In some environments it is not straightforward to estimate a value function but straightforward to represent an optimal policy. Imagine slippery walk that is comprised of n states where n is big; around the middle of the walk the best policy would just to head all the way to the goal, but representing Q here requires approximating all the n's.

Now, the goal with Policy Gradients method is to maximize a `performance objective`, in contrast with value-based methods which is to learn the value function. We call them `policy gradients` because we want to parametrize a policy and know how to change the parameters in order to maximize expected return.
In general, we can use any policies here but it is easier to limit ourselves to *differentiable policies* so that it is computationally easier to improve them. In David Silver lectures he mentioned there are numerical methods to calculate gradients of nondifferentiable policies but let's just say here we limit ourselves to differentiable ones.

The general workflow is:
```
                                evaluate current policy
                                            |
                                            |
                                            v
                            compute the performance objective
                                            |
                                            |
                                            v
    compute the gradient of the performance objective with respect to the policy parameters
                                            |
                                            |
                                            v
            update the policy parameters proportional to this gradient
```

**Specific Algorithms Covered**
- `REINFORCE` - policy gradient method that uses total returns (as in MC-style) weighted by the gradient of log policy.
- `Vanilla Policy Gradient (VPG)` or `REINFORCE with baseline`: Estimating both policy and value so that we can use the value to compute the advantage function. Further, we also added `negative weighted entropy` into the loss function to encourage the training to learn unevenly distributed actions (unambiguous policy).
- `Asynchronous Actor Critic (A3C)`: To reduce variance, A3c uses n-step returns with bootstrapping to learn the policy and value function, and uses *concurrent actors* to asynchronously generate a broad set of experience samples in parallel with multiple worker-learners for each environment. *Hogwild!* was also introduced here, which is a method to parallelize the update from each environment's learners to the global policy and V function asynchronously without locks (Niu et. al 2011).
- `Generalized Advantage Estimation (GAE)`: similar to A2C but uses TD(lambda)-like advantage but for advantages.
- `Advantage Actor-Critic (A2C)`: The major change from A3C is that instead of having multiple learners (models), we have multiple actors that generates the experience but only a single learner. **Workers still run in parallel**.
Another change is to share weights of some layers between the Policy and Value network. 

**Progression**
```
REINFORCE
    |
    | + GAE + Parallel async workers
    v
   A3C
    |
    | + synchronize the workers, single learner
    v
   A2C
    |
    | + fix wasteful data throwing after 1 update
    | + solve variance by clipping policy gradient
    v
   PPO
```

### Chapter 12

Here we discuss state-of-the-art algorithms as of the present.

**Specific Algorithms Covered**
- `Deep Deterministic Policy Gradient (DDPG)`: DQN but introduces a policy network that yields a continuous action space. Uses deterministic policy s->a but uses Gaussian noise to the deterministic action to encourage exploration.
- `Twin-Delayed DDPG (TD3)`: Adds a double learning technique similar to DDQN; Adds noise to the action passed into the environment but to the target actions. Delays updates to the networks (the twin network updates more frequently).
- `Soft Actor-Critic (SAC)`: Off-policy algorithm but trains a `stochastic policy` as in REINFORCE.
- `Proximal Policy Optimization (PPO)`: An improvement to A2C, but introduces a clipped objective function that prevents the policy from getting too different after an optimization step (like a "coach preventing overreaction"). We use a similar clipping strategy in the value function.
