"""Stub of flint.miles_agent for tests/dryrun_ext.sh: the startup probe only records that it ran."""


def startup_probe(out, train_log, model):
    print(f"[flint probe] start: stub ({train_log})", flush=True)
    return 0
