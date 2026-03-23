# Video Editor: Long-to-Short Video Editing Pipeline

Automated pipeline for editing a long interview/documentary video into a short-form video. Given a raw long video, the system (1) parses quotable segments, (2) generates a first-stage screenplay, (3) retrieves matching video clips for each script segment, and (4) renders the final demo video.

---

## Pipeline Overview

```
Long Video
    │
    ▼
[Stage 1] Video_Processing_Pipeline.ipynb
    │  → interview.json, interview.txt, interview_visuals_random.npy
    │
    ▼
[Stage 2] script_gen inference
    │  → screenplay_summary.csv
    │
    ▼
[Stage 3] Retriever inference (inference_video.py)
    │  → time_interval.csv
    │
    ▼
[Stage 4] array_based_demo.py
    │  → final demo video (.mp4)
    │
    ▼
Short-form video with narration audio + retrieved clips
```

---

## Stage 1: Parse Quotable Segments

**Notebook:** `Video_Processing_Pipeline.ipynb` (recommended via Google Colab)

This notebook transcribes the video, identifies speakers via diarization, detects the narrator, and extracts all non-narrator (interview) segments.

### Quick Start

1. Open `Video_Processing_Pipeline.ipynb` in Google Colab
2. Mount Google Drive and set the video path in the `CONFIG` cell:
   ```python
   VIDEO_PATH = "/path/to/your/video.mp4"
   MODEL_SIZE = "base"        # tiny | base | small | medium | large-v2
   HF_TOKEN   = "hf_..."      # HuggingFace token for pyannote diarization
   ```
3. Run all cells

### Outputs (saved to output directory)

| File | Description |
|------|-------------|
| `interview.json` | Structured segments: `{speaker, text, start, end}` |
| `interview.txt` | Plain text utterances, one per line |
| `interview_visuals_random.npy` | CLIP visual embeddings, shape `[N, 2304]` |
| `report.txt` | Speaker breakdown and statistics |

### Local Alternative

Use `run_test.py` for local execution (configure paths at the top of the file):

```bash
pip install whisperx pyannote.audio librosa
python run_test.py
```

---

## Stage 2: Script Generation

**Directory:** `script_gen/`

Run the script_gen model inference to produce a first-stage screenplay from the interview base. The model generates a `screenplay_summary.csv` with structured narration and speaker segments.

```bash
python script_gen/inference.py \
  --interview_base /path/to/interview_base \
  --output_dir     /path/to/screenplay
```

The output `screenplay_summary.csv` contains columns: `Tag` (`Narration` or `SpeakerID`), `Text`, `Duration`.

---

## Stage 3: Zeroshot Retriever Inference

**Script:** `retriever/inference_video.py` (also at `NoteVLM/src/zeroshot/inference_video.py`)

This stage maps each screenplay segment to a time interval in the original video.

- **Narration rows** → UniVTG temporal grounding (binary-search threshold)
- **SpeakerID rows** → BART infill encoder + FusionModule visual retrieval

### Required Models

| Model | Argument |
|-------|----------|
| UniVTG checkpoint | `--univtg_resume` |
| BART infill model | `--bart_model` |
| FusionModule checkpoint | `--fusion_model` |

### Run Inference

```bash
python retriever/inference_video.py \
  --video_name      <VIDEO_ID> \
  --screenplay_dir  /path/to/screenplay \
  --interview_base  /path/to/interview_base \
  --video_clip_dir  /path/to/video_clips \
  --univtg_resume   /path/to/model_best.ckpt \
  --bart_model      /path/to/bart.pth \
  --fusion_model    /path/to/fusion.pth \
  --output_dir      /path/to/output
```

Optional arguments:
- `--fps` — clip sampling rate (default: `1.0`)
- `--min_duration` — minimum clip duration in seconds (default: `3`)
- `--device` — `cuda` or `cpu` (default: `cuda`)

### Output

`time_interval.csv` — each screenplay row annotated with `start_time`, `end_time`, and `part` (video clip file).

---

## Stage 4: Render Demo Video

**Script:** `utils/array_based_demo.py` (from REGen)

Takes the `time_interval.csv` from Stage 3 and the original video, and stitches together the final short-form video. Narration segments are paired with synthesized narration audio (`chunk_<id>_speech.wav`); non-narration (interview) segments use the original video audio.

- If narration audio is **shorter** than the retrieved video clip → silence is padded to the audio
- If narration audio is **longer** → black frames are appended to the video
- All segments are concatenated in screenplay order

### Required Inputs

| Input | Description |
|-------|-------------|
| `time_interval.csv` | Output of Stage 3 (`tag`, `narration_id`, `start_time`, `end_time`) |
| `main.mp4` | Original long video |
| `chunk_<id>_speech.wav` | Synthesized narration audio files, one per narration segment |

### Usage

```python
from pathlib import Path
from utils.array_based_demo import build_compilation

build_compilation(
    csv_file           = Path("/path/to/time_interval.csv"),
    video_file         = Path("/path/to/main.mp4"),
    narration_audio_dir= Path("/path/to/narration_audio/"),
    output_file        = Path("/path/to/output_demo.mp4")
)
```

Or for batch processing across multiple videos:

```python
from utils.array_based_demo import generate_scale

generate_scale(
    video_name_list = ["<VIDEO_ID>", ...],
    audio_input     = Path("/path/to/narration_audio_root"),
    input_dir       = Path("/path/to/time_interval_root"),
    output_dir      = Path("/path/to/output_root"),
    video_dir       = Path("/path/to/video_root"),
    output_name     = "demo",
    csv_name        = "time_interval"
)
```

The narration audio directory should contain files named `chunk_0_speech.wav`, `chunk_1_speech.wav`, etc., corresponding to `narration_id` values in the CSV.

### Output

A single `.mp4` file with narration segments voiced by synthesized audio and interview segments using original audio, edited in screenplay order.

---

## Directory Structure

```
Video_Editor/
├── Video_Processing_Pipeline.ipynb   # Stage 1: segment extraction
├── run_test.py                        # Stage 1: local alternative
├── script_gen/                        # Stage 2: screenplay generation
├── retriever/
│   └── inference_video.py             # Stage 3: zeroshot retrieval
├── utils/
│   └── array_based_demo.py            # Stage 4: demo video rendering
└── example_output/                    # Sample outputs
```
