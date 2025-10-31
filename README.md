# agent-trajectory-eval-pass-k

> **Era:** 2025+ · Staff AI Engineer focus area (agent evaluation). Technique from 2025+, reimplemented in 2026 with Python 3.12 and the standard library.

## Problem

A single agent run cannot distinguish a reliable tool user from a lucky one. Call accuracy also needs separate measurements for missing calls, extra calls, wrong tool names, and wrong arguments. This repository evaluates deterministic scripted agents over repeated stochastic trajectories, reports micro tool-call precision/recall and exact argument matching, and estimates the probability that **all** `k` independent runs succeed.

No model service, network, or API key is involved. SHA-256 draws make every failure and statistic exactly replayable from the agent name, task id, repetition, and seed.

## Method

### Tool calls

Expected and actual calls are matched one-to-one by tool name. Within each name, exact canonical-JSON argument matches are paired first, which maximizes argument credit for duplicate calls. For `m` name matches, `a` actual calls, `e` expected calls, and `x` exact-argument matches:

```text
precision = m / a        recall = m / e        argument exact match = x / m
```

A trajectory succeeds only when `m = a = e = x`: no missing, extra, wrong-name, or wrong-argument calls. Empty/empty trajectories define precision, recall, and argument exact match as 1; otherwise an empty denominator yields 0. Corpus metrics sum counts before taking ratios (micro averaging).

### pass^k reliability

For `c` successful runs out of `n`, with `1 <= k <= n`, the unbiased estimator of all-`k` success is

```text
pass^k(c,n) = product_{i=0..k-1} (c-i)/(n-i),   and 0 when c < k
```

This is not `pass@k`, which estimates at least one success. For example, `c=2,n=4,k=2` gives `pass^2=1/6` but `pass@2=5/6`.

For each task, `pass_k_moments` computes the exact plug-in variance under `C ~ Binomial(n,p)` with `p=c/n`:

```text
E[pass^k(C,n)] = p^k
Var(pass^k(C,n)) = sum_j Pr(C=j) pass^k(j,n)^2 - p^(2k)
```

The observed estimate and its model variance are separate values. Across `T` independent tasks, the mean's variance is `sum_t Var_t / T^2`; the report includes its standard error, a clipped normal-approximation 95% interval, and descriptive sample variance between task estimates. The tests compare these formulas against independent brute-force binomial summation.

## Results

`results/evaluation.json` contains the full reproducible run: four tasks, 32 repetitions per task, seed 2026, `k=3`, three agents, and zero external API calls.

| Agent | Call precision | Call recall | Argument exact match | Mean pass^1 | Mean pass^3 | pass^3 SE | 95% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| perfect | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | [1.0000, 1.0000] |
| mixed | 0.8953 | 0.8906 | 0.9181 | 0.6484 | 0.4100 | 0.0381 | [0.3353, 0.4847] |
| weak | 0.7824 | 0.7865 | 0.8344 | 0.3125 | 0.0446 | 0.0210 | [0.0035, 0.0857] |

The 29 tests cover missing/extra/wrong-name/wrong-argument calls, duplicate-name matching, JSON key order, empty conventions, estimator edge cases, the `pass^k` versus `pass@k` distinction, exact variance, aggregation, deterministic replay, and all four scripted failure modes. The estimator, variance centering, and matching logic were mutation-checked by deliberately breaking them and confirming that tests fail.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
pytest -q
python scripts/evaluate.py --repetitions 32 --k 3 --seed 2026
```

The script rewrites `results/evaluation.json` deterministically except for its real timestamp.

## Limitations

- Calls are evaluated as multisets, not ordered sequences; argument and final-answer semantics beyond exact JSON equality are not scored. Canonical JSON ignores object key order but preserves array order and numeric representation.
- Scripted failure probabilities create a controlled evaluation harness, not evidence about a production LLM agent. Integrating a real agent requires only producing `Trajectory` objects.
- Per-task variance is exact **conditional on the fitted binomial `p=c/n`**, not distribution-free uncertainty about `p`. Cross-task standard errors assume independent tasks. The 95% interval uses a normal approximation and can be optimistic for few tasks or extreme rates.
- `k` cannot exceed the number of observed runs. All-`k` reliability decreases sharply with `k`, as the results demonstrate.
- Floating-point `math.comb`/binomial summation is intended for practical repeated-evaluation sizes, not arbitrary-precision statistics.

## References

- Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021), for the unbiased sampling-estimator style and the contrasting `pass@k` metric.
- [JSON RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) for the argument data model.
- [SHA-256](https://www.rfc-editor.org/rfc/rfc6234) for deterministic pseudo-random replay draws.
