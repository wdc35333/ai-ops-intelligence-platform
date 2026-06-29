# Case studies — the platform earning its keep

> Sanitized extracts from real production sessions. Kiosk names, IPs, domains, customer data,
> and internal repo names are replaced with placeholders (`Kiosk-A`, `the VM`, `the kiosk-client repo`).
> The engineering — the signals, the root causes, the fixes — is real.

These four stories are the point of the whole repo. Anyone can wire an LLM to a few tools; what
this shows is the platform **catching a real problem**, the **judgment to debug it without trusting
the signal blindly**, and the **restraint to not ship tools the data can't support**.

---

## 1. An agent signal → a two-root-cause, cross-repo bug

**Why this is the headline:** the agent didn't "solve" anything by itself. It pointed a finger at
the right kiosk, a human followed the thread, and the thread led somewhere non-obvious — *two
distinct root causes living in two different repositories.*

**The tool.** I added a read-only tool, `get_double_charge_signals`. The kiosks already run a
local guard that auto-voids a duplicate card charge; when it fires it ships a `PAYMENT_VOID` event
to the central `event_log`. The tool aggregates those per kiosk over a 14-day lookback and flags
any kiosk with an abnormal cluster (`PAYMENT_VOID ≥ 3`). (A companion tool,
`get_payment_consistency`, runs six composite integrity checks: oversell, unswept holds, PG-id
missing on a paid row, refund/timestamp mismatch, membership-sync stuck, orphaned kiosk payment.)

**What the agent reported.** On a real run, one kiosk — call it **Kiosk-A** — lit up with a large
cluster of `PAYMENT_VOID` events. Every other kiosk in the fleet was at or near zero. That single
signal is what started the investigation.

**The forensics.** Pulling Kiosk-A's field operation logs confirmed the first root cause: a flaky
**card terminal** was double-charging customers; the local guard auto-voided most duplicates, but a
subset of the *voids themselves failed* — meaning a handful of customers were genuinely
double-charged and not auto-refunded. Concrete, money-on-the-line, exactly the class of thing an
ops agent should surface.

**The twist — control for the confound.** Before blaming the hardware for everything, I pulled a
**second kiosk's** logs as a control. Both kiosks' log volume had exploded by **~30×** starting on
the *same release date*. That ruled out "Kiosk-A's hardware" as the sole story: a chunk of the
noise was a **fleet-wide logging regression**, not a single bad terminal.

**Second root cause, different repo.** Tracing it into the kiosk-client repo: a serial-bus mutex
refactor had started logging *every* bus transaction. Door-status polling (`waitUntilClosed`,
`pollAllStatus`) calls the bus dozens of times per door, so the refactor multiplied log output
~30× across the fleet the moment it shipped. Fix: a `quiet` flag on the bus-transact path that
suppresses the routine status-poll chatter while keeping real command/error logs. Merged to the
client repo's master with **no version bump** (logging-only change, physical fleet — don't force an
unnecessary firmware roll).

**What it demonstrates**
- An agent's value is often *triage*, not *resolution* — narrowing a fleet to the one node worth a human's time.
- Cross-repo systems debugging: the symptom (central payment events) and the two causes (a field card terminal **and** a client-side logging refactor) lived in three different places.
- **Don't trust a single signal.** The second-kiosk control is what separated the real hardware fault from the logging artifact. Same spike, same date — only the comparison told them apart.

---

## 2. Making drift detection trustworthy on real data

**The symptom.** The first time `/drift` was pointed at real production data, it returned
**`critical` on every single run.** A monitor that always cries wolf is worse than no monitor — it
trains everyone to ignore it.

Debugging it surfaced a chain of textbook MLOps pitfalls, each fixed in `services/ml-anomaly`:

| # | Pitfall | Why it fired | Fix |
|---|---|---|---|
| 1 | **PSI saturation** | PSI pinned at ~11.5 (`= −ln(1e-5)`) — an artifact of `np.histogram` dropping out-of-range values plus a clip floor | Standardize by the **reference median / robust scale** (MAD × 1.4826); fixed **±4σ bins with ±∞ outer edges** so nothing is dropped; **Laplace smoothing** |
| 2 | **Trend ≠ drift** | A slowly rising, uptime-correlated metric looks like a distribution shift to a naïve PSI | **Linear detrend** (least-squares residuals) before computing PSI |
| 3 | **Baseline too short** | Only ~30 h of history existed; PSI on 15h-vs-15h halves is pure noise | **Baseline guard**: if `baseline_hours < 168` (one week), report *"collecting baseline"* and force every severity to `ok` |
| 4 | **Contamination artifact** | IsolationForest with a fixed `contamination` flags ~2% of points *no matter what the data does* | **Operational gating**: only emit an item when a resource crosses a real threshold (disk 80/90%, mem 85/95%) **and** the IF flag coincides with a robust-z ≥ 3.5 |

The baseline guard is the one I'm proudest of because it's about **surfacing uncertainty to the
consumer.** The `/drift` response now carries `sufficient_history` and `baseline_hours`, and the
agent's prompt instructs it: when `sufficient_history=false`, *do not judge drift* — report
"baseline still building" and move on. The model is allowed to say "I don't know yet."

**Result:** a clean `info` report on the real VM, verified end-to-end.

**What it demonstrates** — production anomaly/drift ML is mostly the discipline of **not firing on
artifacts**: robust scaling, detrending, a sufficiency gate, and operational thresholds; plus
propagating uncertainty so the downstream consumer can abstain instead of guessing.

---

## 3. Knowing what *not* to build

A tool surface is a **trust budget.** Every tool that can emit a confident-but-wrong answer taxes
the credibility of *every* report the agent writes. Two deliberate non-additions:

- **`get_device_health` — built, then removed.** It looked useful, but the central DB cannot
  distinguish a real fault from several benign states: a kiosk reverted to legacy firmware, a unit
  delivered-but-not-yet-installed, and a site without internet **all present identically** as
  "offline." A diagnosis tool that can't tell those apart would generate confident false alarms, so
  it was removed rather than shipped.
- **A free-form "operations copilot" (Q&A) — declined.** Evaluated against the operator's actual
  workflow and judged unnecessary; a scheduled, structured report serves the real need better than
  an open chat box nobody would open.

**What it demonstrates** — restraint as an engineering output. Deferring `kiosk-fault-diagnosis`
and removing `device_health` kept the agent's signal-to-noise high. The fact that the remaining
8 tools are all ones the data can *actually* support is the reason a `critical` from this agent is
worth reading.

---

## 4. MLOps deployment — 10% model, 90% operational seams

Getting the MLOps loop *actually running* on a resource-constrained production VM (~3.6 GB RAM):

- **Reused, didn't duplicate.** MLflow tracking + registry deployed alongside the existing
  `ml-anomaly` container (same Docker network), so there's one scoring service, not two.
- **Scheduler chosen for the box, not the brochure.** Airflow standalone wants ~1.5 GB RAM, which
  didn't fit, so the daily scoring job runs on **host cron** hitting `POST /pipeline/run`. The
  Airflow DAGs are in the repo and switch on when there's headroom; the *loop* doesn't depend on
  them.
- **UI without a tunnel.** MLflow has no built-in auth, so exposing its UI meant putting **Basic
  Auth** in front of it via the existing reverse proxy — never a public `0.0.0.0` bind.

Two operational gotchas that cost real time (and are the kind of thing you only learn by shipping):
1. **`docker-compose env_file` interpolates values** — it ate the `$` characters in the bcrypt
   hash, silently corrupting it, so the login just failed with no useful error. Fix: escape `$`→`$$`
   when writing the hash into the env file.
2. **`caddy reload` doesn't re-read `env_file`** — the proxy has to be recreated for a new secret to
   take effect.

**What it demonstrates** — "deploying ML" is mostly the operational seams: fitting the resource
budget, handling secrets through three layers of tooling, and standing up auth in front of a
service that has none.

---

## The through-line

Across all four: the model is a **swappable part**, and the hard, valuable work is everything
around it — choosing what to measure, *not* firing on artifacts, knowing which tools the data can
honestly support, and the operational plumbing that makes any of it run unattended. That's the job
this platform is a portfolio for.
