# Decisions

## D1 - Judge model
- Brief asks for Opus 4.8 as judge.
- Probe result: Opus 4.8 gives AccessDenied in the lab account.
- Decision: probe picked Opus 5.5 automatically. Still a different, stronger model than the worker (Sonnet 4.6), so the judge stays independent.

## D2 - No narrative column
- CFPB API/CSV returns only 15 columns; "Consumer complaint narrative" is missing.
- Impact: triage, cases agent, G1. Waiting for trainer's decision.
