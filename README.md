# EMS Whisper Evaluation Bundle

Run our 3 fine-tuned Whisper models (and a vanilla baseline) on **your** EMS audio and transcripts. You only need to point the script at two paths.

## What's in this bundle

```
ems_whisper_eval_bundle/
├── run_evaluation.py          ← Main entry point. Run this.
├── requirements.txt           ← Python deps (pip install)
├── README.md                  ← You are here
├── models/                    ← LoRA adapters (~64 MB each)
│   ├── v5_single/             ← Fine-tuned on 102 synthetic wavs (VoxCPM raw + radio)
│   ├── v6_single/             ← Fine-tuned on 822 synthetic wavs (8× v5)
│   └── v6_aug/                ← Fine-tuned on 1644 radio-domain-augmented wavs (4× aug)
├── lib/                       ← Internal scripts (you don't need to touch)
│   ├── prepare_dataset.py
│   ├── evaluate.py
│   ├── recompute_wer.py
│   └── ems_eval/              ← EMS abbreviation expansion + medical vocab
└── sample_data/
    └── sample_transcripts.csv ← Format reference
```

The base model `openai/whisper-large-v3` (~3 GB) is downloaded automatically from HuggingFace Hub on first run; not bundled.

## Requirements

- **Python 3.10+** (tested on 3.11)
- **GPU recommended** (whisper-large-v3 needs ~6 GB FP16 / 12 GB FP32). CPU works but each clip takes ~30 s instead of ~3 s.
- ~5 GB free disk for HuggingFace model cache

## Setup

```bash
# 1. (Optional but recommended) create a fresh venv / conda env
python -m venv .venv && source .venv/bin/activate

# 2. Install deps
pip install -r requirements.txt
```

## Usage

You need **two inputs**:

1. **A folder of `.wav` files** (16 kHz mono recommended; the script will resample if needed)
2. **A CSV with columns `Filename, transcript`** — one row per wav. Other columns are ignored. See [`sample_data/sample_transcripts.csv`](sample_data/sample_transcripts.csv) for the exact format.

Then run:

```bash
python run_evaluation.py \
  --wav_dir       /path/to/your/wav_folder \
  --transcript_csv /path/to/your/transcripts.csv
```

Optional flags:
- `--output_dir /custom/results/path` (default: `./results/`)
- `--models baseline v6_single` (subset of models; default: all four)

## What you'll get

Inside `results/` (or your `--output_dir`):

```
results/
├── _test_dataset/               # HuggingFace Dataset cache (auto-cleanable)
├── baseline_predictions.csv     # per-clip: original_file, reference, prediction
├── v5_single_predictions.csv
├── v6_single_predictions.csv
├── v6_aug_predictions.csv
└── summary.csv                  # WER/CER per model under 4 normalization modes
```

`summary.csv` and the console output look like:

```
model           n | raw WER raw CER | ems WER ems CER | whisper WER whisper CER | combined WER combined CER
baseline       50 |  88.3%   75.5% |   82.9%  73.9% |    81.7%   73.3% |    81.9%   73.4%
v5_single      50 |  88.3%   75.6% |   83.0%  74.0% |    81.7%   73.4% |    81.9%   73.5%
v6_single      50 |  89.0%   75.5% |   83.3%  73.6% |    82.3%   73.0% |    82.5%   73.1%
v6_aug         50 |  91.1%   77.0% |   86.8%  75.2% |    85.3%   75.0% |    85.4%   75.0%
```

### Which WER column should I look at?

**`combined` is the fairest** — it normalizes both reference and prediction with:
1. EMS abbreviation expansion (e.g. `pt` → `patient`, `bp` → `blood pressure`)
2. Whisper's `EnglishTextNormalizer`, which canonicalizes digit ↔ word forms (`"eighty-two"` ↔ `"82"`), removes `[x]` markers, expands contractions, and strips punctuation/case.

Why this matters: our v6 training data uses spelled-out numbers (`"eighty-two-year-old"`) while real EMS annotations typically use digits (`"82 year old"`). Without normalization, this format mismatch alone inflates raw WER by ~10 pp.

## Background (our results on n=50 of our own data)

| Model | combined WER | combined CER |
|---|---:|---:|
| baseline (no fine-tune) | **81.9%** ⭐ | 73.4% |
| v5_single | 81.9% | 73.5% |
| v6_single | 82.5% | 73.1% |
| v6_aug | 85.4% | 75.0% |

On our test set of 50 real EMS radio clips, **none of the synthetic-data fine-tunes outperform the baseline**, and the radio-domain audio augmentation actively hurts (−3.5 pp). We'd love to see whether the same pattern holds on your Harvard EMS data, or whether your domain is closer to one of the synthetic configurations.

## Troubleshooting

- **`CUDA out of memory`**: run one model at a time with `--models baseline`, then `--models v6_single`, etc.
- **`No module named 'peft'`**: re-run `pip install -r requirements.txt`. PEFT is required to load LoRA adapters.
- **`boto3.__spec__ is None`**: a `boto3` install is partially broken. Run `pip install --force-reinstall boto3`.
- **WAV files not found**: the script matches CSV `Filename` exactly against the basename of files in `--wav_dir` (no recursion). Check both sides match (e.g. case sensitivity, .wav vs .WAV).

## Reference

Full methodology and per-experiment numbers in our internal report:
`results_v5_v6_baseline_20260505/EMS_V5_V6_BASELINE_RESULTS.md`

Contact: [you@example.com] for questions or to share back results.
