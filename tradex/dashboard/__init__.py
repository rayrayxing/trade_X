"""Read-only dashboard: a local web page over the ledger.

It opens the SQLite ledger read-only, never talks to a broker and never writes. Every number
on it is a ledger row (or a report file the research tools wrote); when the ledger is empty
the pages say so instead of showing placeholders. Install the web dependencies with
``pip install -e '.[dashboard]'`` and start it with ``tradex dashboard``.
"""
