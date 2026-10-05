"""Profit upgrades that live outside the protected paths (profit spec P1-P15, the evaluation half).

- ``exits``        exit engine for trade plans: breakeven, trailing, partial targets, time and volatility exits
- ``calibration``  trade probability and EV in R calibrated on forward results (refuses on thin evidence)
- ``suggest``      pyramiding and vol-targeted compounding as sizing suggestions the risk gate may shrink
- ``costcal``      cost model that learns from realised fills against the model
- ``whatif``       counterfactual ledger reports: what blocked plans would have done, per gate and reason
- ``ruin``         ruin radar: named historical shocks replayed over the open book, expected shortfall and flags

Everything is read-only over the ledger and pure over its inputs. Nothing here sizes, routes or sends an order;
the wiring the core needs is in ``patches/ov-exits/``. Run ``python -m tradex.profit --help`` for the reports.
"""
