"""
Prioritised experience replay, Eq. (16)-(17).

p_i  = |delta_i| + eps_p                      (16)
P(i) = p_i^alpha / sum_k p_k^alpha            (17)
A sum-tree gives O(log N) sampling and priority updates.
"""
import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.cap = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.ptr = 0
        self.size = 0

    def total(self):
        return self.tree[0]

    def update(self, leaf, value):
        i = leaf + self.cap - 1
        change = value - self.tree[i]
        self.tree[i] = value
        while i > 0:
            i = (i - 1) // 2
            self.tree[i] += change

    def add(self, value):
        leaf = self.ptr
        self.update(leaf, value)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)
        return leaf

    def find(self, s):
        i = 0
        while 2 * i + 1 < len(self.tree):
            l = 2 * i + 1
            if s <= self.tree[l]:
                i = l
            else:
                s -= self.tree[l]
                i = l + 1
        return i - (self.cap - 1)


class PrioritizedReplay:
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta0=0.4, eps=1e-3, seed=0):
        self.tree = SumTree(capacity)
        self.alpha, self.beta0, self.eps = alpha, beta0, eps
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros((capacity, action_dim), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.d = np.zeros((capacity, 1), np.float32)
        self.max_p = 1.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.tree.size

    def add(self, s, a, r, s2, d):
        leaf = self.tree.add(self.max_p ** self.alpha)
        self.s[leaf], self.a[leaf], self.r[leaf], self.s2[leaf], self.d[leaf] = s, a, r, s2, d

    def sample(self, batch, progress):
        beta = self.beta0 + (1 - self.beta0) * progress
        seg = self.tree.total() / batch
        idx = np.array([self.tree.find(self.rng.uniform(seg * i, seg * (i + 1))) for i in range(batch)])
        idx = np.clip(idx, 0, self.tree.size - 1)
        pri = self.tree.tree[idx + self.tree.cap - 1]
        prob = pri / self.tree.total()                                 # Eq. (17)
        w = (self.tree.size * prob) ** (-beta)
        w = (w / w.max()).astype(np.float32)[:, None]
        return idx, (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]), w

    def update(self, idx, td):
        p = np.abs(td) + self.eps                                      # Eq. (16)
        self.max_p = max(self.max_p, p.max())
        for i, v in zip(idx, p):
            self.tree.update(int(i), float(v) ** self.alpha)
