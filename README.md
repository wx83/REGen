# REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing

Official implementation for **REGen**, a two-stage framework that:
1) generates a coherent short-video script with quotation placeholders, and  
2) retrieves quotable clips from a long source video to fill those placeholders.

**Paper:** [REGen (arXiv:2505.18880)](https://arxiv.org/abs/2505.18880)  
**Project page:** [Demo website](https://wx83.github.io/REGen/)

## Highlights

- Hybrid abstractive+extractive long-to-short video editing
- Supports quote insertion grounded in source interviews
- Includes text-only and text+visual quote retrievers
- Evaluated with objective and human metrics (coherence, alignment, realism)

## Repository Structure

```text
REGen/
├── llama_finetune/
│   ├── data/
│   │   ├── quote_only/
│   │   └── quote_with_contents/
│   └── script/
│       └── finetune_llama3.py
├── quoteretriever/
│   ├── T/                   # text-only retriever
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── evaluation.py
│   └── TV/                  # text+visual retriever
│       ├── dataset_visual.py
│       ├── model_visual.py
│       └── evaluation_visual.py
├── data_preprocessing/
│   └── audio_sep.py         # ASR + diarization utilities
└── utils/
    ├── construct_interview_base.py
    ├── gpt_tf_queue.py
    ├── generate.py
    ├── generate_textual.py
    └── array_based_demo.py
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core training / retrieval dependencies
pip install torch transformers sentence-transformers datasets peft bitsandbytes accelerate pandas numpy wandb

# Optional utilities used by preprocessing / generation scripts
pip install whisperx librosa pydub moviepy openai matplotlib
```

## Required Local Modules

Some scripts import local helper modules not included on this branch, such as:
- `loader` (commonly expected helpers: `load_txt`, `save_txt`, `load_json`, `save_json`)
- `run_on_video`, `main.config`, and `utils.basic_utils` (used by retrieval/generation integration scripts)

If these are not available in your environment, those scripts will fail to import.  
For reproducible execution, ensure these internal utilities are present before running full pipelines.

## Data Layout (Expected by Retriever/Generation Scripts)

Many scripts assume paths like:

```text
<root>/<first_char>/<video_name>/
├── script_timeline_combined.csv
├── interview.txt
├── interview.json
└── interview_visuals_random.npy   # for TV retriever
```

And a narrator mapping CSV:

```text
<root>/narrator_id_by_length.csv
```

## Training

### Stage 1: Script Generator (LLaMA LoRA)

The CLI entrypoint is `llama_finetune/script/finetune_llama3.py`.

**Quote-only training**
```bash
python llama_finetune/script/finetune_llama3.py \
  llama_finetune/data/quote_only/train_finetune_dataset.csv \
  --output_dir checkpoints/llama_quote_only \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --batch_size 4 \
  --epochs 3 \
  --max_length 512
```

**Quote-with-contents training**
```bash
python llama_finetune/script/finetune_llama3.py \
  llama_finetune/data/quote_with_contents/train_finetune_dataset.csv \
  --output_dir checkpoints/llama_quote_with_contents \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --batch_size 4 \
  --epochs 3 \
  --max_length 512
```

### Stage 2: Quote Retriever

Retriever logic is implemented in:
- text-only: `quoteretriever/T/model.py`
- text+visual: `quoteretriever/TV/model_visual.py`

Current files primarily expose dataset/model/training functions (`train_bart`) for integration from a driver script or notebook.  
Evaluation helpers are in:
- `quoteretriever/T/evaluation.py`
- `quoteretriever/TV/evaluation_visual.py`

## Preprocessing Utilities

- `data_preprocessing/audio_sep.py`: diarization/transcription helpers (WhisperX-based)
- `utils/construct_interview_base.py`: builds interview text/json and visual embedding bases
- `utils/array_based_demo.py`: stitches selected clips and narration into an output video

## Evaluation

We evaluate REGen at three levels: script generation, quote retrieval, and end-to-end teaser quality.

| Metric | What it measures | Why we need it | Better |
| --- | --- | --- | --- |
| QDI (Quotation Density Index) | Average number of inserted quotes per documentary | Checks whether quote frequency is realistic | Target-matching |
| QCR (Quote Coverage Rate) | Percentage of videos with at least one correctly inserted quote | Measures quote insertion coverage | ↑ |
| OR (Overlap Ratio) | Word overlap between generated quotes and matched ground-truth interviews | Measures quote faithfulness | ↑ |
| Recall@1 / @5 / @10 | Top-k quote retrieval correctness | Core retriever ranking quality | ↑ |
| Insertion Effectiveness (human) | How naturally interview clips support nearby claims | Captures end-user usefulness of inserted quotes | ↑ |
| ROUGE-1 / ROUGE-2 / ROUGE-L F1 | Script overlap with reference teasers | Measures lexical and sequence similarity | ↑ |
| G-Eval | LLM-based narrative coherence | Measures story flow beyond n-gram overlap | ↑ |
| F1 (retrieval-based teaser metric) | Selected visual content vs. ground truth | Measures retrieval accuracy in final teaser | ↑ |
| SCR (Scene Change Rate) | Scene transition frequency | Measures pacing/temporal dynamics | Target-matching |
| REP (Repetitiveness) | Repeated content in generated teaser | Penalizes redundant clips | ↓ |
| VTGHLS | Highlight likelihood relative to title/topic | Measures highlight salience | ↑ |
| CLIPS-I / CLIPS-N | Audio-visual alignment for interview/narration segments | Measures multimodal alignment quality | ↑ |
| Interview Ratio | Fraction of teaser duration from interview clips | Tracks narration/interview balance | Target-matching |
| Coherence / Alignment / Realness (human) | Human ratings of overall teaser quality | Captures real viewing quality | ↑ |

## Zeroshot Inference

Please refer to the `zeroshot` branch.

## Citation

```bibtex
@inproceedings{xu2025regen,
  title     = {REGen: Multimodal Retrieval-Embedded Generation for Long-to-Short Video Editing},
  author    = {Xu, Weihan and Ma, Yimeng and Huang, Jingyue and Li, Yang and Ma, Wenye and Berg-Kirkpatrick, Taylor and McAuley, Julian and Liang, Paul Pu and Dong, Hao-Wen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```