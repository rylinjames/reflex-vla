"""Inspect pi0_base's normalization layer class names.

swap_rmsnorm_variants matched 0 layers on pi0 — class names must differ.
Find out what's actually in there.
"""
import sys
import types

for _mod in ("lerobot.policies.groot.groot_n1", "lerobot.policies.groot.modeling_groot"):
    _stub = types.ModuleType(_mod)
    _stub.GrootPolicy = None
    _stub.GR00TN15 = None
    sys.modules[_mod] = _stub

import torch  # noqa
from collections import Counter

from lerobot.policies.pi0.modeling_pi0 import PI0Policy

print("Loading lerobot/pi0_base ...")
policy = PI0Policy.from_pretrained("lerobot/pi0_base")
policy.eval()

# Count all module class names, looking for norm-ish ones
names = Counter()
for _, m in policy.named_modules():
    names[type(m).__name__] += 1

print("\nAll module classes with norm/rms/layer in name (case-insensitive):")
for cls, n in sorted(names.items(), key=lambda x: -x[1]):
    if any(k in cls.lower() for k in ("norm", "rms", "layer")):
        print(f"  {n:4d}  {cls}")

print("\nTop 20 module classes overall:")
for cls, n in names.most_common(20):
    print(f"  {n:4d}  {cls}")

# Sanity: walk a few modules and check `weight` shape on anything with 'norm' in name
print("\nSample norm modules (first 5 with 'norm' in class name):")
found = 0
for name, m in policy.named_modules():
    cls = type(m).__name__
    if "norm" in cls.lower() or "rms" in cls.lower():
        w = getattr(m, "weight", None)
        eps = getattr(m, "variance_epsilon", getattr(m, "eps", None))
        print(f"  {name}  cls={cls}  weight_shape={tuple(w.shape) if w is not None else None}  eps={eps}")
        found += 1
        if found >= 5:
            break
