---
title: 'Salts and Layers: Running Multiple AB Tests Simultaneously'
author: Nikita Podlozhniy
summary: >-
  How MD5-based traffic splitting with salts and a layer system lets you run
  dozens of orthogonal experiments without conflicts
date: '2022-09-13'
format:
  hugo-md:
    output-file: ab-salts.md
    html-math-method: katex
    code-fold: true
jupyter: python3
execute:
  enabled: true
  cache: true
tags:
  - ab-testing
  - traffic-splitting
  - experimentation-platform
categories:
  - posts
---


## Background

A mature product runs dozens of AB tests simultaneously.
Every user may participate in multiple experiments at the same time --- different features, different pages, different flows.
Left unmanaged, experiments can interact with each other in ways that corrupt results: a user in the test arm of experiment A and the test arm of experiment B will experience both changes at once, making it impossible to attribute any observed effect cleanly.

The standard solution is a **layer system** with **salt-based hashing** for traffic allocation.
This post walks through how it works --- from a basic single-experiment split to a three-salt architecture supporting exclusive traffic, re-usable slots, and conflict-free concurrency.

## Basic Traffic Split

The simplest approach: hash the user ID with a salt and take the remainder modulo 2.

``` python
import hashlib

def assign_group(visitor_id: str, salt: str) -> str:
    digest = int(hashlib.md5((visitor_id + salt).encode()).hexdigest(), 16)
    return 'test' if digest % 2 == 1 else 'control'
```

For large traffic, two experiments using **different salts** are **orthogonal**: each group in experiment A contains approximately half test and half control users from experiment B.
Neither experiment biases the other.

In practice, traffic is divided into $2^{16} = 65536$ slots for finer-grained allocation:

<details class="code-fold">
<summary>Basic traffic allocation via MD5</summary>

``` python
import hashlib
import numpy as np


SLOTS = 2 ** 16  # 65 536


def get_slot(visitor_id: str, salt: str) -> int:
    digest = int(hashlib.md5((visitor_id + salt).encode()).hexdigest(), 16)
    return digest % SLOTS


# Verify orthogonality: two experiments with different salts
np.random.seed(42)
users = [str(i) for i in range(10_000)]

slots_a = np.array([get_slot(u, 'experiment_A') for u in users])
slots_b = np.array([get_slot(u, 'experiment_B') for u in users])

in_test_a = slots_a < SLOTS // 2
in_test_b = slots_b < SLOTS // 2

# Among test users of A, what fraction are also in test of B?
overlap = in_test_b[in_test_a].mean()
print(f'Fraction of test-A users also in test-B: {overlap:.3f}  (expected ≈ 0.500)')
```

</details>

    Fraction of test-A users also in test-B: 0.500  (expected ≈ 0.500)

## The Problem with a Single Salt

When two conflicting experiments must use the **same salt** (same traffic pool), users are partitioned once and divided between the experiments:

``` python
group = md5(visitor_id + salt) % 65536
# Experiment 1 uses slots 0–16383
# Experiment 2 uses slots 16384–32767
```

This works while both experiments are live.
The problem arises when experiment 1 **ends**: you cannot reuse its slots for a new experiment.
Users who were in slots 0--16383 during experiment 1 are not a fresh random sample --- they have already been "treated" by experiment 1, and any carryover effect would contaminate the new experiment.

## Two-Salt Solution

Add a second salt (a **shuffle salt**) to re-randomise users within each experiment slot:

``` python
# First: which experiment does this user fall into?
which_exp = md5(visitor_id + layer_salt) % num_experiments

# Second: which group within that experiment?
is_test = md5(visitor_id + shuffle_salts[which_exp]) % 2 == 1
```

where `shuffle_salts` is an array of per-experiment salts.

When experiment 0 ends and a new one launches in its slot, only `shuffle_salts[0]` needs to change.
Users in that slot are re-randomised from scratch --- no carryover from the previous experiment.

<details class="code-fold">
<summary>Two-salt assignment with reusable slots</summary>

``` python
def assign_two_salt(
    visitor_id: str,
    layer_salt: str,
    shuffle_salts: list,
) -> tuple:
    n = len(shuffle_salts)
    which_exp = int(hashlib.md5((visitor_id + layer_salt).encode()).hexdigest(), 16) % n
    is_test = int(hashlib.md5((visitor_id + shuffle_salts[which_exp]).encode()).hexdigest(), 16) % 2 == 1
    return which_exp, 'test' if is_test else 'control'


shuffle_salts = ['salt_exp0_v1', 'salt_exp1_v1', 'salt_exp2_v1']
results = [assign_two_salt(u, 'layer_main', shuffle_salts) for u in users[:5]]
for u, (exp, grp) in zip(users[:5], results):
    print(f'User {u:>5} → experiment {exp}, group {grp}')
```

</details>

    User     0 → experiment 0, group test
    User     1 → experiment 0, group test
    User     2 → experiment 2, group test
    User     3 → experiment 2, group test
    User     4 → experiment 2, group control

## Three-Salt Architecture: Exclusive Traffic

Some experiments require **exclusive** traffic --- users who are not participating in any other experiment.
This is common for experiments that fundamentally change the product experience and would contaminate any concurrent test.

A third salt handles this:

``` python
global_slot  = md5(visitor_id + global_salt)  % 65536
layer_slot   = md5(visitor_id + layer_salt)   % 65536
group_slot   = md5(visitor_id + experiment_salt) % 65536

# First 10% of global slots are exclusive (not in any other experiment)
in_exclusive = global_slot < 6553
```

The three-tier hierarchy:

1.  **Global salt** --- separates the exclusive pool from the regular pool.
2.  **Layer salt** --- assigns non-exclusive users to specific experiments within a layer.
3.  **Experiment salt** --- randomises group assignment (test/control) within the experiment.

<details class="code-fold">
<summary>Three-salt assignment with exclusive pool</summary>

``` python
EXCLUSIVE_THRESHOLD = int(0.10 * SLOTS)  # 10% exclusive


def assign_three_salt(
    visitor_id: str,
    global_salt: str,
    layer_salt: str,
    experiment_salt: str,
    exclusive_threshold: int = EXCLUSIVE_THRESHOLD,
) -> dict:
    def slot(vid, s):
        return int(hashlib.md5((vid + s).encode()).hexdigest(), 16) % SLOTS

    in_exclusive = slot(visitor_id, global_salt) < exclusive_threshold
    layer_group  = slot(visitor_id, layer_salt)
    is_test      = slot(visitor_id, experiment_salt) < SLOTS // 2

    return {
        'exclusive': in_exclusive,
        'layer_slot': layer_group,
        'group': 'test' if is_test else 'control',
    }


# Verify ~10% end up in exclusive pool
assignments = [assign_three_salt(u, 'global_salt_v1', 'layer_salt_v1', 'exp_salt_v1') for u in users]
exclusive_rate = np.mean([a['exclusive'] for a in assignments])
print(f'Fraction in exclusive pool: {exclusive_rate:.3f}  (target ≈ 0.100)')
```

</details>

    Fraction in exclusive pool: 0.106  (target ≈ 0.100)

## Conclusion

Salt-based MD5 hashing provides a simple, deterministic, and stateless traffic allocation mechanism that scales to any number of simultaneous experiments:

| Architecture | Experiments | Reusable slots | Exclusive pool |
|----|----|----|----|
| Single salt | Multiple (non-conflicting) | No | No |
| Two salts (layer + shuffle) | Multiple (conflicting OK) | Yes | No |
| Three salts (global + layer + experiment) | Multiple | Yes | Yes |

The key invariant: **as long as salts are distinct and drawn independently, experiments are orthogonal** --- each user's assignment in one experiment is independent of their assignment in any other.
This allows reliable causal inference even when the same user participates in many concurrent experiments.
