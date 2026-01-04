# Notes

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
- `TD(lambda)`: Use the concept of **eligibility traces** where we keep track of states where we've been, then update V according to these traces. Mathematically it's like lambda-weighing n-step TD's.

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
