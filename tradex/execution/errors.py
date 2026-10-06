"""Typed venue errors. The core catches any adapter exception and writes a Health fault
(ok=False, sent loud); these types say which kind, so alerts and runbooks can tell an
outage from a refusal. None of them carries a secret or an account ID in its message."""
from __future__ import annotations


class VenueError(RuntimeError):
    """Anything a venue adapter could not do."""


class VenueUnavailable(VenueError):
    """Timeout, connection reset, 5xx or rate limit: the venue may be fine a moment later."""


class VenueAuthError(VenueError):
    """401/403 or a locked trade context: needs Ray (token, login), not a retry."""


class OrderRejected(VenueError):
    """The venue refused the order (insufficient margin, market closed, bad price...)."""


class OrderUncertain(VenueError):
    """A submit timed out and the client-ID lookup could not say whether it landed. The
    order is NOT resubmitted; the next attempt with the same client ID looks it up first."""


class ClientIdCollision(VenueError):
    """The client order ID already exists at the venue for a different order: a reused
    decision ID. Never treated as "already submitted"."""


class WrongEnvironment(VenueError):
    """A live host, a REAL trade environment or a non-paper account: refused outright."""
