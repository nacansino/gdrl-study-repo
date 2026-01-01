# Gym → Gymnasium Notebook Migration Rules

These rules were deduced by comparing an original Jupyter notebook with its migrated version.
They can be applied systematically to future notebooks.

---

## 1. Replace `gym` with `gymnasium` (keep alias as `gym`)

**Before**
```python
import gym
```

**After**
```python
import gymnasium as gym
```

If multiple imports exist:
```python
import gymnasium as gym, gym_walk, gym_aima
```

---

## 2. Replace `env.seed(seed)` with `env.reset(seed=seed)`

Gymnasium removes `env.seed()` in favor of seeding via `reset()`.

**Before**
```python
env.seed(123)
```

**After**
```python
env.reset(seed=123)
```

When combined with NumPy / Python seeding:
```python
random.seed(seed)
np.random.seed(seed)
env.reset(seed=seed)
```

---

## 3. Unpack `env.reset()` return values

Gymnasium’s `reset()` returns `(observation, info)`.

**Before**
```python
state = env.reset()
```

**After**
```python
state, _ = env.reset()
```

If used inline:
```python
state, _ = env.reset()
done = False
```

---

## 4. Update `env.step()` to the 5-value return signature

Gymnasium’s `step()` returns:

```
(observation, reward, terminated, truncated, info)
```

**Before**
```python
next_state, reward, done, info = env.step(action)
```

**After**
```python
next_state, reward, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

If reward is unused:
```python
state, _, terminated, truncated, info = env.step(action)
done = terminated or truncated
```

---

## 5. Replace `env.env` with `env.unwrapped`

Access to environment internals is now done via `unwrapped`.

**Before**
```python
P = env.env.P
```

**After**
```python
P = env.unwrapped.P
```

---

## 6. Replace deprecated NumPy dtype `np.object`

NumPy removed `np.object`.

**Before**
```python
np.array(data, np.object)
```

**After**
```python
np.array(data, object)
```

---

## 7. Update deprecated environment IDs

Some environment IDs were bumped.

**Example**

**Before**
```python
gym.make("FrozenLake-v0")
```

**After**
```python
gym.make("FrozenLake-v1")
```

Always verify environment IDs when migrating.

---

## Quick Migration Checklist

- [ ] `gym` → `gymnasium as gym`
- [ ] `env.seed(x)` → `env.reset(seed=x)`
- [ ] `obs = env.reset()` → `obs, _ = env.reset()`
- [ ] `step()` unpack → handle `terminated` and `truncated`
- [ ] `done = terminated or truncated`
- [ ] `env.env` → `env.unwrapped`
- [ ] `np.object` → `object`
- [ ] Verify env IDs (`v0` → `v1`)

---

**Scope:** These rules match exactly the transformations observed in the migrated notebook.
