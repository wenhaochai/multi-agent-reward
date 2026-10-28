"""Teams of Recursive Flint agents in a long-horizon game on Frontier-CS (see README.md)."""


class Aborted(Exception):
    """The episode cannot go on (no model server answers, the harness's git fails): the driver records an aborted row
    and a later run retries the episode."""
