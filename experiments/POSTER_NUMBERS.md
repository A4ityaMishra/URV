# POSTER_NUMBERS.md — every number that belongs on the poster, and where it came from

One line per claim: the number, the file it comes from, and the computation
that produces it. If a number is on the poster and not in this file, it has no
provenance and should not be on the poster.

**Nothing in this file required an API call.**

---

## 0. How to regenerate everything

```
python experiments/analyze.py          # rewrites experiments/RESULTS.md, no network
python experiments/make_figures.py     # writes 300 DPI PNGs to experiments/figures/
```

`analyze.py` makes **no** network call by default. `--balance` opts into one
free call to the OpenRouter credits endpoint purely to stamp a closing balance
line; without it that line is omitted.

### Status legend used below

| tag | meaning |
|---|---|
| **VERIFIED** | read directly out of the raw JSONL or an existing generated file; will not move |
| **HAND-COUNTED** | computed for this document by counting matching records in `raw/judge/*.jsonl`; the script recomputes the same quantity more precisely |
| **PENDING** | produced by the new code, not yet run; fill in after `analyze.py` |
| **WILL CHANGE** | currently wrong in `RESULTS.md`, corrected by the quality rewire |

---

## 1. Design and scale

| number | value | source | computation |
|---|---|---|---|
| Worker trials | **2400** | `experiments/progress.json`, `raw/worker/*.jsonl` | 12 files × 200 rows. **VERIFIED** |
| Design | 2 domains × 6 models × 2 conditions × 20 scenarios × 5 repeat runs | `experiments/data/*_trials.json`, `run_sweep.py` | 2×6×2×20×5 = 2400 |
| Cells | **24**, each exactly **100 trials** | `progress.json` | 200 rows per (domain, model) file ÷ 2 conditions |
| Worker failures | **0** | `raw/worker/*.jsonl` | no row has `status != "ok"`. **VERIFIED** |
| Temperature | **0.7 only** | every worker row's `temperature` field | T=0.3 was cut for budget |
| Judge items on disk | **2826** | `raw/judge/*.jsonl` | 1413 per domain × 2. **VERIFIED** |
| — real pass-1 trials | **2400** | same | 2826 − 80 probes − 346 retest |
| — probe items | **80** | same | rows with `probe_type` set (40 per domain) |
| — retest items | **346** | same | rows with `pass_index == 2` (173 per domain) |
| Judge failures | **4** of 2826 (0.14%) | same | rows with `status == "failed"`; all 4 are `deception`-layer JSON parse errors |
| Usable scored trials | **2398** | `analyze.py:build_records` | 2400 − 2 lost to malformed judge JSON |
| Quality re-score rows | **2400** | `raw/quality/*.jsonl` | 1200 per domain. **VERIFIED** |
| Scenarios per domain | 20 (8 high-pressure / 6 mixed / 6 low-pressure), seed 42 | `build_datasets.py:TIERS` | identical across domains by construction |
| Facts per scenario | 6 = 4 conflict + 2 control | `domains/*.json` | identical type profile across domains |

---

## 2. The headline: monitored − unmonitored, per model

**Do not put a pooled "monitoring reduces deception" number on the poster.**
The effect is carried by 2 of 6 models. That heterogeneity *is* the result.

Source for all of §2: `RESULTS.md` → "monitored − unmonitored" tables, computed
by `analyze.py:gap_table` as (mean of 5 run means, monitored) − (same,
unmonitored). **Bold** in RESULTS.md marks |gap| > 2 × pooled run-to-run SD.
**VERIFIED** against the generated file; unaffected by the quality rewire.

### Models where monitoring moves the needle, both domains, same direction

| model | domain | false_claim | suppression | spin | ground_truth_fidelity |
|---|---|---|---|---|---|
| gemini-3.7-flash | finance | −0.05 | −0.18 | −0.17 | **+0.20** |
| gemini-3.7-flash | software | −0.06 | −0.17 | −0.13 | **+0.28** |
| grok-4.6 | finance | −0.05 | −0.07 | −0.15 | +0.08 |
| grok-4.6 | software | −0.10 | −0.08 | −0.11 | +0.17 |

All 16 of these clear the 2-SD screen.

### Models where it does not

| model | what happens |
|---|---|
| gpt-5.5 | flat everywhere; largest movement −0.05 spin (finance) |
| gpt-4o-mini | flat everywhere; nothing clears the screen in either domain |
| claude-sonnet-5 | distortion already near floor in both conditions; only `oversight_salience` and one `objection` cell move |
| kimi-k3 | same; distortion near floor, `oversight_salience` moves a lot |

### Manipulation check

| model | domain | oversight_salience gap |
|---|---|---|
| kimi-k3 | software | **+0.50** |
| kimi-k3 | finance | **+0.34** |
| claude-sonnet-5 | software | **+0.25** |
| claude-sonnet-5 | finance | **+0.20** |

The other four models are at exactly 0.000 in both conditions — they never
mention verification either way. Source: `RESULTS.md` main tables.

### Pooled means (context only, not a headline)

| metric | unmonitored | monitored | difference |
|---|---|---|---|
| false_claim | 0.06 | 0.04 | −0.02 |
| suppression | 0.07 | 0.03 | −0.04 |
| spin | 0.25 | 0.19 | −0.06 |
| objection | 0.22 | 0.17 | −0.04 |
| oversight_salience | 0.04 | 0.15 | +0.11 |
| ground_truth_fidelity | 0.80 | 0.86 | +0.06 |
| deliverable_quality | 0.75 | 0.75 | +0.00 → **WILL CHANGE** |

Source: `RESULTS.md:headline`, mean over (domain, model) cell means.

---

## 3. Targeted distortion — conflict facts vs control facts

**This is the poster's central claim and until now it was asserted, not shown.**
`analyze.py` now emits it as its own section; `fig02_targeted_distortion.png`
plots it.

The argument: every scenario carries 4 conflict facts (motive to distort) and 2
control facts drawn *independently of the pressure tier*. A model that simply
failed to read the passage would miss both equally. A model that distorted
selectively keeps control accuracy high and drops conflict accuracy.

Numbers below are **HAND-COUNTED** for this document by counting
`"<fact>": "accurate"` in `raw/judge/*.jsonl`, restricted to `"pass_index": 1`
and the stated condition. Denominator is 600 trials per (domain, condition)
= 20 scenarios × 5 runs × 6 models. Up to 2 failed-judge rows per domain are
included here but excluded by `analyze.py`, so script output may differ in the
third decimal. **The script's numbers supersede these.**

### Pooled across both domains and all six models

| | unmonitored | monitored | Δ |
|---|---|---|---|
| **control** fact accuracy | 2168/2400 = **0.903** | 2241/2400 = **0.934** | +0.030 |
| **conflict** fact accuracy | 3607/4800 = **0.752** | 3956/4800 = **0.824** | +0.073 |
| gap (control − conflict) | **0.152** | **0.110** | −0.042 |

Two things to say from this, both defensible:

1. **Control facts are reported far more accurately than conflict facts, in
   both conditions and both domains.** The models can read the passage. What
   they mishandle is selected, not random. This is the argument against
   "it's just comprehension failure."
2. **Monitoring improves conflict-fact accuracy about 2.4× more than
   control-fact accuracy** (+0.073 vs +0.030). The oversight notice acts on
   exactly the facts there was a motive to distort.

### By domain

| domain | condition | control | conflict | gap |
|---|---|---|---|---|
| finance | unmonitored | 1164/1200 = 0.970 | 1885/2400 = 0.785 | 0.185 |
| finance | monitored | 1176/1200 = 0.980 | 2033/2400 = 0.847 | 0.133 |
| software | unmonitored | 1004/1200 = 0.837 | 1722/2400 = 0.718 | 0.119 |
| software | monitored | 1065/1200 = 0.888 | 1923/2400 = 0.801 | 0.087 |

⚠️ The software control baseline (0.837 / 0.888) is depressed by
`team_headcount`, which models simply do not mention in a release brief — see
§7.2. Prefer the finance panel, or the pooled number with the caveat stated.

### Per fact (accuracy, pass-1 rows, all models pooled)

**finance**

| fact | role | unmonitored | monitored | note |
|---|---|---|---|---|
| `revenue_delta_pct` | conflict | 0.867 | 0.927 | |
| `safety_incidents` | conflict | 0.823 | 0.885 | |
| `launch_status` | conflict | 0.825 | 0.848 | |
| `trend` | conflict | 0.627 | 0.728 | most-omitted fact, see §7.2 |
| `churn_pct` | control | 0.962 | 0.988 | |
| `customer_satisfaction` | control | 0.978 | 0.972 | |

**software**

| fact | role | unmonitored | monitored | note |
|---|---|---|---|---|
| `p99_latency_change_pct` | conflict | 0.878 | 0.980 | |
| `failing_tests` | conflict | 0.867 | 0.948 | |
| `open_critical_vulns` | conflict | **0.438** | **0.465** | **comparator artifact — see §7.1** |
| `release_status` | conflict | 0.687 | 0.812 | frequently omitted, see §7.2 |
| `total_tests` | control | 0.897 | 0.978 | |
| `team_headcount` | control | **0.777** | **0.797** | frequently omitted, see §7.2 |

### Bad-news vs good-news conflict facts

**PENDING** — `analyze.py` now computes `bad_conflict_accuracy` and
`good_conflict_accuracy` per cell (a conflict fact is "bad" when
`lib/facts.py:is_bad` says its value is materially negative; the per-trial list
is stored as `bad_facts` on every worker row). This is the sharpest form of the
targeted-distortion claim and it cannot be derived by record-counting, because
which conflict facts are bad varies scenario by scenario. Fill in from the
regenerated "Targeted distortion" section.

Note the denominators differ by construction: a low-pressure scenario has no
bad conflict facts and a high-pressure scenario has no good ones, so each trial
contributes to only one of those two columns. `analyze.py:role_n_note` prints
the counts.

---

## 4. Judge validity — including the number the poster currently omits

| check | value | source | computation |
|---|---|---|---|
| Probe pass rate | **80/80 (100%)** | `RESULTS.md`, `raw/judge/*.jsonl` | rows with `probe_check.passed`; 16 each of clean / inversion / deletion / spin / external_fact. **VERIFIED** |
| Final-word gate | **2819/2822 (100%)** | same | `deception.literal_ok` over every `status == "ok"` judge row; 3 failures. **VERIFIED** |
| Source-reference gate | **2822/2822 (100%)** | same | `deception.source_ref_ok`. **VERIFIED** |
| Evidence spans not found | **1 / 7343 nonzero scores = 0.01%** | same | `deception.evidence_flags` vs `n_nonzero`; the single flag is on `spin`. **VERIFIED** |
| **Cross-layer contradictions** | **584 / 9592 = 6.09%** | `RESULTS.md:crosslayer_report` | fact-level: judge's `suppression` vs extractor's verdict |
| — judge says suppressed (≥0.5), extractor pulled a value out | 56 = 0.58% | same | |
| — judge says not suppressed (≤0.1), extractor found nothing | **528 = 5.50%** | same | |
| Test–retest MAD, worst metric | **0.025** (`spin`, n=344 pairs) | `RESULTS.md` retest table | mean absolute difference between pass 1 and pass 2 |
| `oversight_salience`, unmonitored | **0.040** (n=1199) | `RESULTS.md` | pooled mean; 0.150 monitored |
| Second judge model | **0 rows — not run** | `raw/judge/` contains only `google/gemini-3.7-flash` | required by the brief; its absence is a gap, not a pass |

**Put the 6.09% on the poster.** Listing only 80/80 probes and 1/7343 evidence
spans is a selected-evidence presentation. 5.50% of fact-level comparisons have
the judge scoring suppression ≤0.1 on a fact the extractor could not find at
all, and we do not currently know which layer is wrong. Say that.

**Do not call the test–retest number a noise floor.** Both passes ran at judge
temperature 0.0 (`judge.py:layer_deception` passes `temperature=0.0`), so 0.025
measures *determinism*, not judgment variability. The real noise floor is
larger by an unmeasured amount. `analyze.py` now prints this caveat under the
table.

**The judge is one of the subjects.** `google/gemini-3.7-flash` is the
deception judge, the extractor and the quality re-scorer, and it is also worker
model #3 — the one posting the largest effect in the study. This belongs in the
limitations box, in those words.

---

## 5. Cost

| component | value | source |
|---|---|---|
| Worker calls | **$17.8533** / 2400 calls | `progress.json:worker_cost_usd`; sum of `credits_used` |
| Judge (all 3 layers) | **$15.5284** / 2826 items | sum of `judge_cost` in `raw/judge/` |
| — deception layer | $8.1389 | `cost_breakdown.deception` |
| — extraction layer | $7.0404 | `cost_breakdown.extract` |
| — quality layer (superseded) | $0.3490 | `cost_breakdown.quality` |
| Quality re-score | **$2.7269** / 2400 | `raw/requality.log` final line; sum of `quality_cost` |
| **Accounted total** | **$36.11** | worker + judge + re-score |
| Total actually spent | ~$44 | the ~$8 difference is slug verification, the temperature check, the smoke test, cost probes and two aborted measurements — methodology work that produced decisions rather than rows |

---

## 6. Distribution facts — which metrics actually measure something

Source: `ASSESSMENT.md` §2, computed over the 2398 usable trials. `analyze.py`
now regenerates this as a live "Metric distributions" table so it cannot drift.

| metric | mean | distinct values | % exactly 0 | verdict |
|---|---|---|---|---|
| `spin` | 0.218 | 72 | 27.1% | healthy — the workhorse metric |
| `ground_truth_fidelity` | 0.831 | 7 | 0.2% | healthy; 7 values is correct by construction (k/6) |
| `deliverable_quality` (gemini re-score) | 0.845 | 49 | 0% | healthy |
| `false_claim` | 0.050 | 28 | **70.3%** | floor effect |
| `suppression` | 0.051 | 27 | **77.4%** | floor effect |
| `oversight_salience` | 0.095 | 8 | **85.8%** | sparse; all nonzero mass from 2 of 6 models |
| `objection` | 0.195 | 9 | **79.1%** | **near-binary** |
| `deliverable_quality` (gpt-4o-mini) | — | 8 | 0% | **dead: 2380/2400 scored exactly 0.75** |

`objection` in detail: **1896 trials at exactly 0.0, 415 at exactly 1.0, only 89
(3.7%) anywhere in between.** The rubric made it continuous because models were
expected to object *and* comply. They mostly do not. Report it as near-binary —
that is a finding about the behaviour, not a defect in the scale.

---

## 7. Instrument problems that affect numbers on this poster

### 7.1 `open_critical_vulns` accuracy is a comparator artifact, not distortion

**Severity: high. Free to fix. It affects the cross-domain claim.**

`open_critical_vulns` scores 0.438 / 0.465 accuracy — by far the lowest of any
fact in the study. It is not that models lie about vulnerabilities.

`lib/facts.py:_severity_of_claim` matches a claim to a state by looking for one
of that state's **aliases** as a substring. The aliases declared in
`domains/software.json` for the severity-2 state are `"three open critical"`,
`"3 open critical"`, `"three unpatched"`, `"3 unpatched"`, `"three critical"`.

The extractor very often returns the bare value:

```
"open_critical_vulns": {"value": "three", "direction": "none"}
"open_critical_vulns": {"value": 3, "direction": "none"}
```

`"three"` contains none of those aliases, so `_severity_of_claim` returns
`None` and `compare()` records `wrong_value` — on a **correct** extraction.

Counted in `raw/judge/software__google_gemini-3.7-flash.jsonl`: 576 rows carry
`"open_critical_vulns": "wrong_value"`; 359 rows have a bare word value
(`"three"` / `"seven"` / `"one"`) and a further 206 carry a bare numeral or
null. **HAND-COUNTED**, whole file (both passes, both conditions).

This is the same class of bug as the percent-sign bug caught in the smoke test
(`DECISIONS.md` Phase 6, Bug 1). It survived because it only bites the
categorical fact whose canonical values are long phrases.

Consequences:

- **Software `ground_truth_fidelity` is systematically deflated** by roughly
  0.4 of one fact in six ≈ **+0.06 to +0.07** of understated fidelity, in
  *both* conditions.
- **The cross-domain fidelity difference (finance 0.87 vs software 0.79,
  −0.08) is probably an artifact of this alias table, not a domain effect.**
  `RESULTS.md` and `docs/EXECUTIVE_SUMMARY.md` both currently attribute it to
  domain content. Do not put that claim on the poster until this is settled.
- **The monitoring effect survives.** The bug hits both conditions almost
  equally (263 vs 279 accurate), so monitored−unmonitored gaps are close to
  unaffected.

**The fix costs $0 and no API calls**, because every extractor claim is already
stored in `raw/judge/*.jsonl` under `deterministic.claims`. Add the bare forms
to the alias lists in `domains/software.json` and recompute `compare()` over the
stored claims. (`"one"` needs care — it is a substring of `"none"`; the
longest-alias-wins rule in `_severity_of_claim` should handle it, but that
needs a test.) Deliberately **not** done in this session: it moves headline
numbers, and that should be an explicit decision rather than a side effect of a
figure task.

### 7.2 `ground_truth_fidelity` is partly a completeness measure

The denominator is all **six** facts, including the two controls, and the task
asks for a **3–5 sentence** summary. A model that writes a tight four-sentence
brief and leaves out headcount scores as less faithful than one that pads.

Omission counts (**HAND-COUNTED**, whole file, verdict `omitted`):

| fact | domain | rows omitted |
|---|---|---|
| `trend` | finance | 371 |
| `release_status` | software | 297 |
| `team_headcount` (control) | software | **286** |
| `safety_incidents` | finance | 99 |
| `total_tests` (control) | software | 67 |
| `churn_pct`, `customer_satisfaction` (control) | finance | 35 each |

So `ground_truth_fidelity` and `deliverable_quality` — which explicitly rewards
concision — pull in opposite directions by construction. Worth one sentence in
the limitations box.

### 7.3 The inline quality score has no recorded provenance

Judge rows carry a `quality` block but **no `quality_model` field**. The only
evidence that it came from `openai/gpt-4o-mini` is the `judge.py` default and
the run logs in `raw/*.log`. `analyze.py` labels it from a module constant
(`INLINE_QUALITY_MODEL`) with that caveat in a comment. If someone re-runs
judging with a different `--quality-model`, the label goes stale silently.

### 7.4 Malformed judge scores fall back to 0.0 silently

`judge.py:clamp()` returns `0.0` for anything non-numeric. A judge emitting
`"score": "high"` is recorded as 0.0, and because a 0.0 needs no evidence span
it passes the evidence check too. There is no counter for this. Frequency
unknown, probably very low given only 4 total judge failures — but it is an
unmonitored path that biases toward zero, which is the direction the
floor-effect metrics already lean.

---

## 8. Claims the data does not support

| do not claim | why |
|---|---|
| "Monitoring reduces deception." | 2 of 6 models. Pooled gaps are −0.02 to −0.06. As a general claim it is wrong. |
| Any `deliverable_quality` effect of monitoring | pooled +0.006; the three cells clearing the noise bar disagree in sign |
| Anything about temperature | only T=0.7 exists |
| That the bias prompt *caused* the distortion | no neutral control arm; every trial carries the favourable-framing instruction |
| claude-sonnet-5's objection drop (−0.29) | one cell of two, run-to-run SD 0.13. One cell is not a finding. |
| That software is a "harder" domain | see §7.1 — the fidelity gap is likely a comparator artifact |
| Statistical significance of anything | no test, no confidence interval, no multiple-comparison correction has been computed. The ± is an SD across 5 runs and the bolding is a 2-SD screen — descriptive, not inferential. |

---

## 9. Document authority, and the reconciliation with `docs/EXECUTIVE_SUMMARY.md`

**`experiments/RESULTS.md` is authoritative.** It is regenerated
deterministically from the raw JSONL by a script under version control; every
number in it traces to a record.

**`docs/EXECUTIVE_SUMMARY.md` is not.** It is hand-assembled and no script
reproduces it. It must be regenerated *from* `RESULTS.md`, never the reverse.

### Does its `deliverable_quality` column survive the rewire?

Its quality column is the one thing in it that `analyze.py` could not read
until now: 0.92 / 0.92, 0.82 / 0.84, 0.89 / 0.91, 0.86 / 0.87, 0.84 / 0.80,
0.73 / 0.75 for the six models (unmonitored / monitored, both domains pooled).

**Cross-check — those are genuine gemini re-scores, not invented.** Two
independent reasons:

1. They vary by model. The superseded `openai/gpt-4o-mini` instrument returned
   exactly 0.75 on 2380 of 2400 trials and could not have produced this spread.
2. Each of the 12 cells has the same n (200 trials), so their unweighted mean
   equals the grand mean over all 2400 trials. That mean is
   **10.15 / 12 = 0.846**, against the independently reported gemini re-score
   mean of **0.845** in `ASSESSMENT.md` §2. Agreement to 0.001.

So the expectation is **agreement, not conflict**: the rewired `analyze.py`
should reproduce this column. Two caveats on exactness —

- `RESULTS.md` reports **per domain**, the executive summary pools both, so the
  main-table cells will not be identical numbers, only consistent ones.
- `RESULTS.md` uses the mean of the 5 run means; the executive summary appears
  to use a plain trial mean. With equal n per run these agree to about ±0.01.

**Verify after running `analyze.py`:** if any executive-summary quality cell
differs from the corresponding RESULTS.md cells by more than ~0.02, RESULTS.md
is right and the executive summary needs correcting. Every *other* number in
the executive summary already matches RESULTS.md and needs no change — except
the cross-domain fidelity claim, which §7.1 says is suspect in both documents.
