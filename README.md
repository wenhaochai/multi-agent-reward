# labs-molt team study on miles (Della)

The labs-molt Qwen3.8-27B team game (lead / teammates / solo, LoRA RL) run on radixark/miles (SGLang rollout +
Megatron LoRA through the bridge), next to the molt arms. Ledger: labs-molt `logs/miles_q38.yaml`.

## Setup (once)

- **Image**: `bash build.sh` on della-vis1 builds `miles.sif` from `docker://radixark/miles:latest` (sandbox +
  mksquashfs: a >25 GB pull writes a truncated sif) and clones the miles source; the source is checked out at the
  commit inside the image (`/root/miles`, 874b3e275 for the 2026-09-30 image).
- **opentelemetry fix**: that image ships opentelemetry-api 1.45.0 with sdk/exporters 1.44.0; Ray's dashboard agent
  dies on import and `ray start` times out. `pyfix/` holds api 1.44.0, put first on PYTHONPATH by the launcher:
  `pip download --no-deps opentelemetry-api==1.44.0 -d wheels && pip install --no-deps --target pyfix wheels/*.whl`
  (inside the sif, on a vis node: compute nodes are offline).
- **Data**: `data/` from labs-molt `data/prep_q38mid` (`tools/make_team_data.py` adds `metadata.messages`).

## Launch

- `sbatch_miles.sh` -> `run_miles_in_container.sh`: a job-private Ray head (job-derived ports; miles' own launcher
  `pkill`s sglang/ray, unsafe on shared nodes), env exported before `ray start`, `python3 train.py`.
- `submit_smoke_single.sh`: single-agent GRPO LoRA smoke. `ARM=lead|nob|solo [SMOKE=1] bash submit_team.sh`: the
  team game (`miles_team/`), smoke or full arm (resumable: `--load` = `--save`).

## Tests

- `tests/test_team_rollout.py team|solo` (miles.sif): the team rollout against a fake SGLang.
- `tests/test_core_parity.py` (molt.sif): the copied game core equals molt's.
