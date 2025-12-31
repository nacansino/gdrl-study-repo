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