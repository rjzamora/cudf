# NBBO/VWAP OrderScheme Experiment Learnings

## Summary

The NBBO/VWAP experiment gives us a concrete reason to keep investing in
OrderScheme-based execution: preserving known input order can reduce peak
memory enough to make larger workloads feasible on fewer GPUs.

The most important observation is not only wall-clock speed. With default
single-process streaming settings, the 4-day path can run if the final output
sort is disabled, but the physical final `ORDER BY (ts_bucket, RIC)` is still a
major memory pressure point.

An early version of the experiment incorrectly allowed sortedness on a strict
`ts_bucket` prefix to satisfy `ORDER BY (ts_bucket, RIC)`. That was wrong:
strict `ts_bucket` boundaries do not imply `RIC` ordering within each bucket.
The corrected fast path is to avoid the global sort and sort locally within
each strict `ts_bucket` range. Raw `DateTime` sortedness still helps the
downstream order-aware path, and the final `(ts_bucket, RIC)` ordering can be
completed with much less memory than a full global sort.

## Current Benchmark Signal

The benchmark branch now has two useful knobs:

- `--hint-sorted {none,datetime,datetime-ric}` attaches a sortedness hint at the
  raw parquet scan.
- `--no-output-sort` removes the final `ORDER BY (ts_bucket, RIC)` so we can
  isolate the cost of downstream execution from the final global sort.

Representative 4-day results from the experimental branch:

| Configuration | Result |
|---|---|
| Single process, baseline, final output sort enabled | OOM / high memory pressure in physical final sort |
| Single process, hint-sorted, final output sort enabled | Completed with strict-prefix local final sort, about 22s |
| Single process, baseline, `--no-output-sort` | Completed, about 26-27s |
| Single process, hint-sorted, `--no-output-sort` | Completed, about 19-20s |
| 4-rank Dask, `--no-output-sort` | Baseline and hint-sorted were roughly tied |

The 4-rank timings are noisy because worker startup, cache warmth, sink timing,
and persistent actors matter. However, traces still showed that ordered
execution substantially reduced groupby evaluator traffic. The wall-clock
benefit is partly hidden by scan, expression evaluation, sink, and scheduling
overlap.

## Interpretation

The clean customer-facing motivation is:

> If the input dataset is already sorted, cudf-polars should preserve and use
> that order to avoid memory-heavy repartitioning and global sorting.

This is stronger than "sort-aware execution is faster." For NBBO/VWAP, the
ordered path can reduce downstream redistribution pressure. It also changes the
final ordering problem: a strict prefix does not prove the full requested order,
but it does prove that only local sorting is needed for the remaining suffix
keys.

## What Should Change In Open PRs

Keep the active PRs narrow and composable.

### RapidsMPF OrderScheme work

The RapidsMPF-side work should focus on the metadata primitive:

- boundary storage and retrieval,
- strict vs non-strict boundary metadata,
- cheap boundary construction from known sources when possible,
- enough API surface for cudf-polars to inspect and adjust order boundaries.

Avoid encoding NBBO/VWAP-specific assumptions in RapidsMPF. The benchmark is
motivation, not the abstraction.

### cudf-polars OrderScheme and adjustment work

The cudf-polars PRs should focus on:

- carrying `OrderScheme` through channel metadata,
- conservative propagation and invalidation rules,
- `adjust_orderscheme` as a metadata-agnostic data utility,
- tests for empty owned partitions, strictness requirements, prefix key
  compatibility, and missing collective IDs.

The benchmark's old `early_sort` experiment should not influence the PR design.
The preferred user story is not "sort early." It is "data is already sorted;
tell the engine; let the engine preserve and exploit that fact."

## Features Needed On Main

### 1. Scan-level sortedness hints

`LazyFrame.set_sorted(...)` / hint-sorted support should become a metadata
producer. The clean first target is a hint applied immediately after a parquet
scan, for example raw `DateTime` or `(DateTime, RIC)`.

For parquet scans, footer statistics may let us collect boundary metadata
without reading the sorted column as a separate data pass.

### 2. Monotonic transform propagation

NBBO/VWAP groups by `ts_bucket`, which is derived from `DateTime`. The engine
needs to propagate ordering through monotonic truncation:

```text
DateTime sorted ascending -> ts_bucket sorted ascending
```

This propagation may need to downgrade strictness. Truncation can collapse
distinct timestamp boundary values into the same bucket value, creating repeated
or non-strict boundaries.

### 3. Order-aware groupby

If data is already partitioned by a compatible flat `OrderScheme`, groupby
should use `adjust_orderscheme` or a no-op metadata path instead of falling back
to hash shuffling.

### 4. Order-aware join

The first useful join optimization can be modest:

- if both sides have compatible ordered partitioning, avoid global hash
  repartitioning;
- if one side needs boundary adjustment, use `adjust_orderscheme` to match the
  reference side;
- defer sorting one unsorted side to a later optimization.

### 5. Final sort elision

If metadata proves that the output is already ordered by the requested sort
keys, the final `Sort` should become a pass-through or metadata-only operation.
For `ORDER BY (ts_bucket, RIC)`, metadata on `ts_bucket` alone is not enough.
However, strict `ts_bucket` partitioning is enough to avoid a global sort and
perform a local sort within each existing range.

Possible paths are:

- the dataset is physically sorted by `(ts_bucket, RIC)` or an equivalent
  precomputed key order, in which case final sort can be elided;
- an upstream sort-based groupby/join implementation emits data ordered by the
  full output sort keys, in which case final sort can be elided;
- metadata proves strict partitioning on a prefix of the requested keys, in
  which case the final sort can avoid global repartitioning and sort locally;
- the benchmark/query does not semantically require ordered output and can use
  `--no-output-sort`.

### 6. Canonical partitioning shape

The first pass should prefer:

```text
Partitioning(inter_rank=flat_order_scheme, local="inherit")
```

This avoids early complexity around nested inter-rank/local order schemes.
`HashScheme` with local `OrderScheme` may still be useful later, but it is not
needed to explain the current NBBO/VWAP win.

## Correctness Notes

`adjust_orderscheme` must emit one data message for every locally owned output
boundary range, even when that range has no rows. Downstream actors rely on
`local_count` messages arriving; skipping empty owned partitions risks hangs or
metadata/data count mismatches.

The utility should remain metadata-agnostic. The caller should own input
metadata consumption and output metadata publication. `adjust_orderscheme`
should only transform `TableChunk` messages.

## Suggested Benchmark Story

Keep the benchmark surface small:

```text
--hint-sorted none
--hint-sorted datetime
--hint-sorted datetime-ric
--no-output-sort
```

Report a small matrix:

| Case | Purpose |
|---|---|
| baseline | current production-style behavior |
| hint-sorted | ordered execution end-to-end |
| baseline + `--no-output-sort` | downstream cost without final global sort |
| hint-sorted + `--no-output-sort` | ordered downstream cost without final sort |

This separates two claims:

1. ordered metadata can reduce downstream repartition/groupby traffic;
2. the final `ORDER BY` is a separate memory bottleneck that can be downgraded
   from a global sort to a local sort when strict prefix order is available.

Both claims matter for the customer workflow.
