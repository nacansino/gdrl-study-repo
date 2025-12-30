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