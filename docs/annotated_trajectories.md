# Annotated Trajectories

Two full agent runs, each captured as a structured `reasoning_trace`
(see `docs/trajectories/annotated_success.json` and
`docs/trajectories/annotated_failure.json` for the raw JSON). Both were
produced by `agent/orchestrator.py` after the dialect-detection and
dialect-aware-response fixes described in `WRITEUP.md`.

## 1. Success case (`docs/trajectories/annotated_success.json`)

**Message (MSA, order-status):**
> "وين طلبي رقم ORD-1001؟ ما وصلني لحد الحين"
> ("Where is my order ORD-1001? It hasn't arrived yet")

**What happened:**
- `extract_entities` correctly pulls `ORD-1001` out of the free-text message.
- `classify_intent` (LLM) returns `order_status` with confidence 0.95 — matches the gold label.
- `lookup_order` finds the order in `data/orders.json` (status `shipped`, tracking number `TRK-55421`, expected delivery `2026-06-03`).
- `detect_dialect` correctly returns `msa`.
- `assess_urgency` → `low`, `route_team` → `logistics` — both match gold.
- `draft_response` produces a polite MSA reply that references tracking via email/app, consistent with the detected dialect and intent.
- Total latency ≈ 15.3s, cost ≈ $0.00019 (two LLM calls: classify + draft).

**Why I'm proud of it:** every step of the pipeline — entity extraction,
intent, dialect, order lookup, routing, and the drafted reply — agrees with
the gold label and is internally consistent (the reply actually uses the
order's real tracking number and status from the mock DB, not a generic
template). This is the "thin slice" working end-to-end as designed.

## 2. Failure case (`docs/trajectories/annotated_failure.json`)

**Message (Gulf dialect, long hotel review):**
> "ضعيف جداً... كنت حاجز غرفة مطله على البحر اطلاله كامله .. فقلوا لي انت
> حاجز اطلاله جزئيه ... ضريبه المدينة ركزوا عليها فيها استغلال من الفندق
> ... التامين لديهم يستغلونه في الثلاجه ..." (long complaint about a
> downgraded room, hidden "city tax", and a misused refundable deposit)

**Gold labels:** `expected_intent=payment_issue`,
`expected_team=escalations_management`, `expected_dialect=gulf`.

**What the agent produced:** `intent=complaint`, `team=customer_care`,
`dialect=gulf` (dialect is correct).

**What happened and why it's a failure:**
- `detect_dialect` correctly identifies `gulf` (this case used to be
  misclassified as `msa` before the dialect-detector fix — it now works).
- `draft_response` correctly switches to a Gulf-register reply ("حياك الله...
  تراني بتابع معك..."), not generic MSA — the register-matching fix is working.
- However, `analyze_sentiment` scores the message as **positive** (0.33)
  because the lexicon-based analyzer picks up isolated positive words
  ("رائع", "حلو") embedded in an otherwise scathing review, without weighing
  the dominant negative content. This pushes `assess_urgency` to `low`.
- `classify_intent` (LLM) labels the message `complaint` — a defensible
  reading of a long hotel review, but the gold label `payment_issue` was
  assigned because the core grievance is about being charged hidden fees
  ("ضريبة المدينة", deposit/insurance misuse). The LLM picked the broader
  "general complaint" framing over the narrower financial-dispute framing.
- As a result, the case is routed to `customer_care` (low urgency) instead
  of `escalations_management`, understating the priority of what is, at its
  core, a billing dispute.

**Root cause / takeaway:** this is the same failure mode the error-analysis
breakdown shows is the largest contributor to intent errors
(`payment_issue -> complaint`, `damaged_product -> complaint`,
`shipping_delay -> complaint` in `logs/evaluations/error_analysis.json`):
long, multi-topic hotel reviews (from the HARD dataset) don't map cleanly
onto a single e-commerce intent, and both the sentiment lexicon and the
intent classifier default to the broadest "complaint" bucket for messages
with multiple grievances. Fixing this would require either (a) a
multi-label intent scheme for these long reviews, or (b) a more detailed
intent-classification prompt that explicitly asks the model to identify the
*primary actionable* grievance (billing vs. service vs. logistics) rather
than the overall tone.
