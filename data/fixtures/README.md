This repository keeps only small processed regression fixtures under `data/processed/`.

Policy:

- tracked fixtures are meant for local non-live snapshot/backtest/report flows
- raw market downloads are not tracked
- generated reports, snapshots, logs, and SQLite files are not tracked
- MT5 live workflows are expected to regenerate their own runtime artifacts
- richer demo state should be created with `var-project seed-demo`, not committed back into the repo

The default tracked fixture set targets:

- symbols: `EURUSD`, `USDJPY`
- timeframe: `H1`
- history window tag: `60d`
