"""Runtime pieces around the trading core: bar store, FX rate sources, incremental
signals, the bar-close scheduler and multi-venue books. The same pieces drive replay
and paper/live, so both write the same decision rows."""
