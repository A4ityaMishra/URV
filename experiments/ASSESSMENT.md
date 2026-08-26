# ASSESSMENT — what we actually have, and whether to spend the last $9.67

Read-only audit of `experiments/raw/` as it stands. No API calls were made to
produce this. Balance stated by you: **$9.67 remaining, ~$44 spent.**

Accounted spend visible in the raw files is **$36.11** (worker $17.85 + judge
$15.53 + quality re-score $2.73). The ~$8 gap is slug verification, the
temperature check, the smoke test, extractor and judge cost probes, and the
two aborted cost measurements — i.e. methodology work that produced decisions
rather than rows.

---

## 1. INVENTORY

### Worker data — complete

| | |
|---|---|
| Trials on disk | **2400 / 2400** |
| Failures | **0** |
| Cells (domain × model × condition) | **24**, every one at exactly **100 trials** |
| Trials per cell | 20 scenarios × 5 repeat runs = 100 |
| Domains | 2 (`finance`, `software`) |
| Models | 6 (gpt-5.5, claude-sonnet-5, gemini-3.7-flash, grok-4.6, kimi-k3, gpt-4o-mini) |
| Conditions | 2 (unmonitored, monitored) |
| Repeat runs | 5 (run_index 1–5) |
| Temperatures | **1** (T=0.7 only) |
| Cost | $17.85 |

### Judge data — complete for one judge

| | |
|---|---|
| Real trials scored, pass 1 | **2400** (2398 usable; 2 lost to malformed judge JSON) |
| Test–retest rows, pass 2 | **346** (344 usable pairs) |
| Probe items | **80** |
| Judge failures | 4 of 2826 (0.14%) |
| Judge models used | **1** (`google/gemini-3.7-flash`) |
| Cost | $15.53 |

### Quality re-score — complete

`deliverable_quality` was re-scored for all **2400/2400** trials with
`google/gemini-3.7-flash` ($2.73), because the original `openai/gpt-4o-mini`
quality layer was degenerate (below). Both versions are on disk.

### Validity checks: 5 of 6 done

| brief § | check | status | result |
|---|---|---|---|
| 3a | Probe trials | **DONE** | **80/80 passed** (16 each of clean, inversion, deletion, spin, external_fact) |
| 3b | Anchored continuous rubric | **DONE** | 72 distinct values on `spin`; the scale is being used, not snapped to anchors |
| 3c | Test–retest | **DONE** | 344 pairs; worst-metric MAD **0.0249** |
| 3d | **Two independent judge models** | **MISSING — 0 rows** | no second judge has ever been run |
| 3e | Evidence spans verified | **DONE** | **1 flagged / 7343 nonzero = 0.01%** |
| 3f | Cross-layer contradictions | **DONE** | 9592 fact comparisons, **6.09%** contradiction rate |

**Also missing:** any T=0.3 arm, any neutral/no-bias control condition, any
human-labelled validation of the judge.

### Two things are broken right now and cost $0 to fix

1. **`analyze.py` does not read `raw/quality/`.** The re-scored quality data
   exists but is not wired in, so the current `RESULTS.md` still prints the
   dead gpt-4o-mini column (`0.75 ± 0.00` in all 12 rows).
2. **`RESULTS.md` is stale.** It was generated before the retest and quality
   re-score finished. It says the noise floor is "0.000" and "no test-retest
   pairs on disk". Both are now false. Regenerating is free.

---

## 2. SUFFICIENCY AGAINST YOUR MENTOR'S REQUIREMENTS

| requirement | status | detail |
|---|---|---|
| Single table, one row per (model, agent type) | **MET** | 12 rows per domain, 24 total. Exactly the shape he drew. |
| 5–7 metrics | **MET** | 7. |
| 2–3 capturing deception | **MET** | `false_claim`, `suppression`, `spin`. |
| One measuring task performance | **MET, but only after the re-score is wired in** | `deliverable_quality` (+ `ground_truth_fidelity` as a second performance axis). |
| Metrics generalize beyond the task | **MET** | Rubric names no domain concept; facts arrive as (name, type, label, value). The software domain was added as a data file with zero Python changes — that is the proof. |
| ~40 trials per configuration | **EXCEEDED** | 100 per (domain, model, condition); 200 if pooled across domains. |
| Report what we see, don't tune | **MET** | Nulls are reported as nulls; two smoke-test bugs and one dead metric are documented rather than hidden. |
| One task is not enough | **MET** | Two domains, structurally identical by construction. |

### Which metrics actually measure something

Judged on distribution across 2398 trials:

| metric | mean | sd | distinct values | % exactly 0 | verdict |
|---|---|---|---|---|---|
| `spin` | 0.218 | 0.230 | 72 | 27.1% | **Healthy.** The workhorse metric. |
| `deliverable_quality` (gemini) | 0.845 | 0.119 | 49 | 0% | **Healthy** — but only in the re-scored file. |
| `ground_truth_fidelity` | 0.831 | 0.187 | 7 | 0.2% | **Healthy.** 7 values is correct by construction (k/6). |
| `false_claim` | 0.050 | 0.092 | 28 | **70.3%** | **Weak — floor effect.** Frontier models almost never state outright falsehoods. |
| `suppression` | 0.051 | 0.107 | 27 | **77.4%** | **Weak — floor effect.** |
| `oversight_salience` | 0.095 | 0.253 | 8 | **85.8%** | **Sparse.** 100% of nonzero mass comes from 2 of 6 models. |
| `objection` | 0.195 | 0.386 | 9 | **79.1%** | **DEGENERATE as a continuous metric.** |
| `deliverable_quality` (gpt-4o-mini, superseded) | — | — | 8 | 0% | **DEAD. 2380 of 2400 trials scored exactly 0.75.** |

**`objection` is effectively binary and you should say so.** 1896 trials at
exactly 0.0, 415 at exactly 1.0, and only **89 trials (3.7%)** anywhere in
between. The brief specifically demanded this metric be continuous because
"models frequently object *and* comply." The data says that middle ground
barely exists — models either comply silently or refuse the framing outright.
That is a finding, not a bug, but do not present `objection` as a continuous
scale when 96% of its mass sits on two points.

**Bottom line: yes, this is enough for a poster** — provided you wire in the
quality re-score and regenerate the table first. You have more trials per cell
than he asked for, across two domains rather than one.

---

## 3. WHAT THE DATA ACTUALLY SHOWS

`*` below = the gap exceeds twice the pooled run-to-run standard deviation
across the 5 repeats. Anything unstarred is noise.

### Defensible finding 1 — the effect is model-specific, and that is the headline

Monitoring reduces distortion in **2 of 6 models**, in the same direction, in
**both domains**:

| model | domain | false_claim | suppression | spin | ground_truth_fidelity |
|---|---|---|---|---|---|
| gemini-3.7-flash | finance | −0.054* | −0.175* | −0.169* | **+0.203*** |
| gemini-3.7-flash | software | −0.064* | −0.172* | −0.130* | **+0.278*** |
| grok-4.6 | finance | −0.052* | −0.065* | −0.152* | +0.083* |
| grok-4.6 | software | −0.103* | −0.078* | −0.114* | +0.173* |

Every other model is flat. gpt-5.5 and gpt-4o-mini show nothing anywhere.
This directly answers the question your mentor posed — *"maybe for GPT the
monitored agent is doing worse, but maybe for Grok it is not the case"* — and
the answer is that the monitoring effect does not generalize across models.
That is a better poster result than a uniform effect would have been.

### Defensible finding 2 — the manipulation check works, but only for talkative models

`oversight_salience` rises under monitoring in exactly the models that write
meta-commentary: claude-sonnet-5 (+0.204* finance, +0.253* software) and
kimi-k3 (+0.342*, +0.505*). Zero movement in the other four, because those four
never mention verification in either condition.

**The unmonitored-contamination check passes.** Pooled unmonitored
`oversight_salience` is 0.040. All 126 nonzero unmonitored rows come from
claude-sonnet-5 (20.5% of its rows) and kimi-k3 (42.5%); the other four models
are at exactly 0.000. All 126 have evidence spans that **verify present in the
response text** — e.g. *"if the board later discovers the framing was managed
rather than straight"*, *"misrepresent material facts to a board that needs
accurate information"*. The judge is reading real accountability language, not
inferring the condition. If it were inferring, all six models would show it.

### Defensible finding 3 — dissociation between objecting and distorting

claude-sonnet-5 and kimi-k3 object loudly (objection ≈ 0.76 and 0.59
unmonitored) and distort very little in either condition. gemini-3.7-flash and
grok-4.6 never object (0.00) and distort substantially when unmonitored.
Objecting and distorting are close to independent.

### Defensible finding 4 — a modest domain effect

Software has lower `ground_truth_fidelity` (0.794 vs 0.869) and *lower*
false_claim/suppression than finance. Since the domains are structurally
identical by construction, this is attributable to domain content.

### DO NOT CLAIM

- **"Monitoring reduces deception."** Pooled differences are small
  (`spin` −0.056, `suppression` −0.039, `false_claim` −0.022) and are driven
  entirely by 2 of 6 models. Stated as a general claim it is wrong.
- **Any `deliverable_quality` effect of monitoring.** Pooled +0.006. Three
  cells clear the noise bar but disagree in sign (gemini +0.036, kimi −0.073).
  This is a null.
- **Anything about temperature.** Only T=0.7 exists.
- **Any causal claim about the bias prompt.** There is no neutral control.
- **claude-sonnet-5's objection drop** (−0.289* finance) as a real effect. It
  appears in one cell of two, with a large run-to-run sd (0.13). One cell is
  not a finding.

### What a skeptical reviewer attacks first

1. **The judge is also one of the subjects.** `google/gemini-3.7-flash` is the
   deception judge, the extractor, *and* worker model #3 — and it shows the
   single largest monitoring effect in the study. A reviewer will see this in
   thirty seconds. (Section 6.)
2. **No neutral control condition.** Every trial carries the favourable-framing
   instruction, so you cannot attribute distortion to it.
3. **Test–retest was run at judge temperature 0.0.** MAD 0.0249 measures how
   deterministic the judge is, not how noisy its judgments are. It is an
   unrealistically flattering number and you should present it as a
   *determinism* check, not a noise floor.
4. **`objection` is binary in practice** (§2).
5. **5.50% of fact comparisons have the judge scoring suppression ≤0.1 on a
   fact the extractor could not find at all.** 528 of 9592. Either the
   extractor is over-reporting omissions or the judge is under-scoring
   suppression, and we do not currently know which.
6. **Floor effects on two of the three deception metrics** (70–77% zeros).

---

## 4. MARGINAL VALUE OF SPENDING MORE

| option | cost | what it changes | worth it? |
|---|---|---|---|
| **Finish quality re-score** | **$0 — already complete** | Nothing to buy. It needs ~10 lines in `analyze.py` to be *used*. | **Do the free code fix. Mandatory.** |
| **Finish test–retest** | **$0 — already complete** | 344 pairs is plenty. Re-running at nonzero judge temperature would be more honest, but it changes no headline number. | **No.** |
| **Second judge model** | ~$2.50–4 | The only missing required check (§3d), and the only affordable way to test whether the gemini-judged gemini result survives an independent judge. Changes attack #1 from unanswered to answered. | **YES — the highest-value dollar available.** |
| **T=0.3 arm** | ~$5.65 (finance only, 2 runs) | Buys one weakly-powered stability sentence, with ±  from 2 runs instead of 5. Would consume most of the balance. | **No.** |
| **Third domain** | ~$14 | Unaffordable. And the mentor's "one task is not enough" bar is already cleared with two. | **No.** |
| **More scenarios per domain** | ~$3–6 | You already have 100 trials/cell against a 40-trial requirement. Scenario sampling error is not what limits this study. | **No.** |

**"Stop and write it up" is nearly the right answer.** The dataset is complete
and internally consistent; the marginal experiment adds little. The one
exception is the second judge, because it is not more data — it is the answer
to the question that will otherwise sink the poster.

---

## 5. RECOMMENDATION

**Spend about $4. Keep about $5.60 in reserve. Then stop.**

**Step 0 — free, do first (blocking).** Wire `analyze.py` to read
`raw/quality/*.jsonl`, preferring the gemini score, and regenerate
`RESULTS.md`. Until this is done your main table has a dead column and a false
"noise floor 0.000" line.

**Step 1 — ~$0.05.** Run probes only against one candidate second judge that is
**not one of the six subject models** (gpt-4o and claude-sonnet-4 both appear in
the legacy `outputs/`, so their slugs are known good). 5 probe items is enough
to see whether it is usable, and to price it properly.

**Step 2 — ~$2.50–3.50.** Run that judge over a **targeted** subset: the
`gemini-3.7-flash` and `grok-4.6` rows, both conditions, both domains, ~20
trials per cell (≈160 items) plus a probe set. Do not sample uniformly — the
whole point is to re-check the two cells where the effect lives, with a judge
that did not write the text and is not one of the subjects.

Concretely: if the second judge reproduces the direction and rough magnitude of
gemini's +0.20/+0.28 fidelity gap, finding 1 is defensible and you can say so
on the poster. If it does not, you have discovered that your headline result is
judge-dependent — which you need to know *before* printing it.

**Do not spend on:** T=0.3, a third domain, more scenarios, re-running quality,
re-running test–retest, or scoring the full 2400 with a second judge.

**Then stop and write.** You have 2400 clean trials, zero worker failures,
80/80 probes passing, a 0.01% evidence-span flag rate, and a genuine
model-heterogeneity result. That is a solid undergraduate poster.

---

## 6. BIGGEST WEAKNESS

**The primary judge is also one of the six models being judged — and it is the
model showing the largest effect.**

`google/gemini-3.7-flash` scores every trial's deception metrics, performs every
extraction behind `ground_truth_fidelity`, and re-scored every
`deliverable_quality`. It is simultaneously worker model #3. Its own
unmonitored→monitored fidelity gain (+0.203 finance, +0.278 software) is the
largest single number in the study, and it was measured by itself.

There is no evidence in the data that this is *causing* the result — the effect
also appears in grok-4.6, judged by the same judge, and the probes all pass —
but "we cannot rule it out" is not a position you want to defend at a poster
session.

**Can money fix it?** *Partially, and cheaply.* ~$3 buys an independent judge
over the gemini and grok cells, which converts "we cannot rule it out" into
"we checked, and here is what happened." That is why it is the recommended
spend.

**What money cannot fix:** a fully clean design needs the judge to be outside
the subject pool for *every* number in the table, which means re-judging all
2400 trials with a non-subject model (~$13+ at the rates measured, i.e. more
than the remaining balance), or dropping gemini-3.7-flash from the model set
and losing a cell. Both are design decisions for the next round, not something
to buy today.

**Runner-up weakness, unfixable with money: no neutral control condition.** You
can compare monitored against unmonitored, but you cannot say the
favourable-framing instruction caused any distortion, because there is no arm
without it. State this in the limitations box on the poster before a reviewer
states it for you. Adding that arm is the single highest-value experiment for
the next round.
