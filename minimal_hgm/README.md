# Minimal HyperGraph Memory Pipeline

This is a minimal runnable pipeline for:

1. Training a tiny action LLM policy (`WRITE_SHORT`, `RETRIEVE`, `CONSOLIDATE`)
2. Executing actions on a hypergraph memory graph
3. Managing short-term to long-term memory transition
4. Running a full demo episode end-to-end

## Environment

Recommended conda env (already prepared in this workspace):

```bash
conda activate hgm_verl
```

## Run

```bash
cd /root/HGM_mew/HGM
python minimal_hgm/pipeline_minilm.py --device auto
```

If you want a specific GPU:

```bash
python minimal_hgm/pipeline_minilm.py --device cuda:3
```

## Outputs

- `output/minimal_hgm/tiny_policy_classifier.pt`: trained tiny policy model
- `output/minimal_hgm/run_config.json`: runtime config
- `output/minimal_hgm/demo_trace.json`: full demo trace (actions + memory stats)

## Verl + LoCoMo One-Dialogue RL Run

Download official LoCoMo-10 (10 dialogues):

```bash
mkdir -p data
env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY -u ALL_PROXY \
  curl -L --max-time 60 -s \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
  -o data/locomo10.json
```

Run RL training on one dialogue (index 0):

```bash
python minimal_hgm/verl_locomo_one_dialog_rl.py \
  --locomo-path data/locomo10.json \
  --dialogue-index 0 \
  --device cuda:3
```

Outputs:

- `output/verl_locomo_one_dialog/policy_locomo_one_dialog.pt`
- `output/verl_locomo_one_dialog/train_log.json`
- `output/verl_locomo_one_dialog/qa_subset.json`
