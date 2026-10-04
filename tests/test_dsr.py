"""Deflated and probabilistic Sharpe checked against the papers' own worked examples.

Sources (read from the PDFs at davidhbailey.com, 4 Oct 2026):
- Bailey, D. H. and Lopez de Prado, M. (2014), "The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management
  40(5), section "A numerical example", pp. 9-10 of the SSRN version.
- Bailey, D. H. and Lopez de Prado, M. (2012), "The Sharpe Ratio Efficient Frontier",
  Journal of Risk 15(2): section 3 PSR example (Figure 6 statistics, p. 10) and section 5
  MinTRL examples (p. 11).
"""
import math

import numpy as np
import pandas as pd
import pytest

from tradex.backtest import metrics

DAYS = 250  # both papers annualise daily figures with 250 observations per year


# --- 2014 DSR paper ------------------------------------------------------------------------
# Inputs: N = 100 trials, V[{SR_n}] = 1/2 (annualised), T = 1250 daily returns,
# skew = -3, kurtosis = 10, annualised SR = 2.5.
# Published: SR0 ~= 0.1132 (non-annualised) and DSR = 0.9004 < 0.95.
# With N = 46 trials "DSR would have been 0.9505"; with Normal returns (skew 0, kurt 3)
# "DSR = 0.9505 after N=88 independent trials".

def test_dsr_paper_expected_max_sharpe():
    assert metrics.expected_max_sharpe(100, 0.5 / DAYS) == pytest.approx(0.1132, abs=5e-5)


@pytest.mark.parametrize("n, skew, kurt, expected", [(100, -3, 10, 0.9004), (46, -3, 10, 0.9505), (88, 0, 3, 0.9505)])
def test_dsr_paper_worked_example(n, skew, kurt, expected):
    sr = 2.5 / math.sqrt(DAYS)
    dsr = metrics.dsr_from_moments(sr, 1250, skew, kurt, n, 0.5 / DAYS)
    assert dsr == pytest.approx(expected, abs=5e-5)


# --- 2012 Sharpe Ratio Efficient Frontier paper ---------------------------------------------
# Figure 6, a hedge fund's 2-year monthly track record: SR = 0.458 (monthly), skew = -2.448,
# kurt = 10.164. Published: PSR(0) = 0.982 assuming Normality, 0.913 with the measured
# skew and kurtosis, and 0.953 with 3 years instead of 2. The paper prints three decimals and
# appears to truncate (the formula gives 0.9535 for the last), so the tolerance is 1e-3.

@pytest.mark.parametrize("t, skew, kurt, expected", [(24, 0, 3, 0.982), (24, -2.448, 10.164, 0.913),
                                                    (36, -2.448, 10.164, 0.953)])
def test_psr_paper_hedge_fund_example(t, skew, kurt, expected):
    assert metrics.psr_from_moments(0.458, 0.0, t, skew, kurt) == pytest.approx(expected, abs=1e-3)


# Section 5: MinTRL in years for an annualised SR of 2 to beat 1 at 95% with IID Normal
# returns: 2.73 (daily), 2.83 (weekly), 3.24 (monthly); 4.99 years for monthly returns with
# the HFR aggregate index's skew -0.72 and kurtosis 5.78 (Brooks and Kat 2002, as quoted).

@pytest.mark.parametrize("freq, skew, kurt, years", [(250, 0, 3, 2.73), (52, 0, 3, 2.83), (12, 0, 3, 3.24),
                                                     (12, -0.72, 5.78, 4.99)])
def test_min_track_record_length_paper(freq, skew, kurt, years):
    n = metrics.min_track_record_length(2 / math.sqrt(freq), 1 / math.sqrt(freq), skew, kurt)
    assert n / freq == pytest.approx(years, abs=0.005)


def test_series_psr_matches_moment_form():
    r = pd.Series(np.random.default_rng(3).normal(0.001, 0.01, 800))
    sr = r.mean() / r.std(ddof=1)
    expected = metrics.psr_from_moments(sr, 0.0, len(r), r.skew(), r.kurt() + 3)
    assert metrics.probabilistic_sharpe(r) == pytest.approx(expected)


def test_dsr_falls_as_trials_grow():
    r = pd.Series(np.random.default_rng(4).normal(0.001, 0.01, 1000))
    vals = [metrics.deflated_sharpe(r, n, 1e-4) for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True)
