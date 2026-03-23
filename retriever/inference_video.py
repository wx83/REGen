"""
Single-video inference for the REGen pipeline.

Given one video and its screenplay_summary.csv, assigns video clip time
intervals to each Narration / SpeakerID segment and writes time_interval.csv.

Narration rows  → UniVTG binary-search threshold (TeaserGen-PT)
SpeakerID rows  → BART infill encoder + FusionModule (visual) retrieval

Usage:
  python inference_video.py \\
    --video_name 1gBwyk4MGG8 \\
    --screenplay_dir  /path/to/screen_play_main_test \\
    --interview_base  /path/to/interview_base \\
    --video_clip_dir  /path/to/video_clip \\
    --univtg_resume   /path/to/model_best.ckpt \\
    --bart_model      /path/to/bart.pth \\
    --fusion_model    /path/to/fusion.pth \\
    --output_dir      /path/to/output
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer

from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array
from run_on_video import vid2clip, txt2clip
from run_on_video import clip as clip_module


# ── FusionModule (matches model_visual.py / generate.py) ─────────────────────

class FusionModule(nn.Module):
    def __init__(self, hidden_dim: int = 768, joint_dim: int = 768):
        super().__init__()
        self.fc1 = nn.Linear(2 * hidden_dim, joint_dim)
        self.fc2 = nn.Linear(joint_dim, joint_dim)

    def forward(self, text_emb, visual_emb):
        x = torch.cat([text_emb, visual_emb], dim=-1)
        x = F.relu(self.fc1(x))
        return F.normalize(self.fc2(x), p=2, dim=-1)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_univtg(resume: str, save_dir: str) -> nn.Module:
    """Load UniVTG model via TestOptions (mirrors gpt_tf_queue.py)."""
    # TestOptions.parse() expects an args namespace with at minimum
    # resume and save_dir, so we patch sys.argv to be minimal.
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0], "--resume", resume, "--save_dir", save_dir]
    cudnn.benchmark = True
    cudnn.deterministic = False
    opt = TestOptions().parse()
    model, _, _, _ = setup_model(opt)
    sys.argv = saved_argv
    return model


def load_bart(bart_path: str, device: str):
    """Load fine-tuned BART + tokenizer."""
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    special_tokens = {"additional_special_tokens": ["<sum>"]}
    tokenizer.add_special_tokens(special_tokens)
    bart.resize_token_embeddings(len(tokenizer))
    bart.load_state_dict(torch.load(bart_path, map_location=device))
    bart.eval()
    return bart.to(device), tokenizer


def load_fusion(fusion_path: str, device: str) -> FusionModule:
    fm = FusionModule().to(device)
    fm.load_state_dict(torch.load(fusion_path, map_location=device))
    fm.eval()
    return fm


# ── UniVTG / saliency helpers ─────────────────────────────────────────────────

def _extract_vid_feat(vid_path: Path, fps: float, clip_model, feat_cache_dir: Path) -> np.ndarray:
    """Extract (or load cached) CLIP video features for one clip."""
    clip_len = 1.0 / fps
    save_path = feat_cache_dir / vid_path.parent.name / vid_path.stem
    save_path.mkdir(parents=True, exist_ok=True)
    cache_file = save_path / "vid.npz"
    if not cache_file.exists():
        return vid2clip(clip_model, str(vid_path), str(save_path), clip_len=clip_len)
    return np.load(cache_file)["features"]


def _vtg_forward(vtg_model, txt: str, vid_path: Path, fps: float,
                 clip_model, feat_cache_dir: Path, txt_save_dir: str,
                 device: str) -> np.ndarray:
    """Single forward pass through UniVTG; returns combined saliency score array."""
    txt_feat = txt2clip(clip_model, txt, txt_save_dir).astype(np.float32)
    vid_feat = _extract_vid_feat(vid_path, fps, clip_model, feat_cache_dir).astype(np.float32)

    vid_t = torch.from_numpy(l2_normalize_np_array(vid_feat))
    txt_t = torch.from_numpy(l2_normalize_np_array(txt_feat))

    ctx_l = vid_t.shape[0]
    tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
    tef_ed = tef_st + 1.0 / ctx_l
    vid_t = torch.cat([vid_t, torch.stack([tef_st, tef_ed], dim=1)], dim=1)

    src_vid      = vid_t.unsqueeze(0).to(device)
    src_txt      = txt_t.unsqueeze(0).to(device)
    src_vid_mask = torch.ones(1, src_vid.shape[1]).to(device)
    src_txt_mask = torch.ones(1, src_txt.shape[1]).to(device)

    vtg_model.eval()
    with torch.no_grad():
        out = vtg_model(src_vid=src_vid, src_txt=src_txt,
                        src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)

    logits   = out["pred_logits"][0].cpu().T.numpy()
    saliency = out["saliency_scores"].cpu().numpy()
    return saliency + logits


def get_saliency_curve(text: str, video_name: str, video_clip_dir: Path,
                       fps: float, vtg_model, clip_model,
                       feat_cache_dir: Path, txt_save_dir: str,
                       device: str) -> np.ndarray:
    folder = video_clip_dir / video_name[0] / video_name
    scores = []
    for mp4 in sorted(folder.glob("*.mp4")):
        s = _vtg_forward(vtg_model, text, mp4, fps, clip_model,
                         feat_cache_dir, txt_save_dir, device)
        scores.append(s.squeeze())
    return np.concatenate(scores)


def _find_consecutive_intervals(indices: list) -> list:
    if not indices:
        return []
    intervals, start = [], indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            intervals.append([start, indices[i - 1]])
            start = indices[i]
    intervals.append([start, indices[-1]])
    return intervals


def _clip_interval_extraction(text, video_name, video_clip_dir, threshold, fps,
                               last, last_last, min_dur, tolerance,
                               vtg_model, clip_model, feat_cache_dir,
                               txt_save_dir, device):
    sal = get_saliency_curve(text, video_name, video_clip_dir, fps, vtg_model,
                             clip_model, feat_cache_dir, txt_save_dir, device).flatten()
    global_track = np.ones(sal.shape[0])
    for ts in last + last_last:
        s = max(ts[0] + tolerance, 0)
        e = min(ts[1] - tolerance, len(global_track) - 1)
        global_track[int(s):int(e) + 1] = 0

    avail    = np.where(global_track == 1)[0]
    sel_idx  = avail[sal[avail] > threshold].tolist()
    consec   = _find_consecutive_intervals(sel_idx)
    filtered = [iv for iv in consec if (iv[1] - iv[0] + 1) >= min_dur]
    total    = sum(iv[1] - iv[0] + 1 for iv in filtered)
    return total, filtered


def binary_search_threshold(text, video_name, video_clip_dir, fps, text_duration,
                             last, last_last, min_dur, tolerance,
                             vtg_model, clip_model, feat_cache_dir,
                             txt_save_dir, device,
                             lo: float = -1.0, hi: float = 1.0,
                             precision: float = 0.001):
    best_thresh = None
    best_dur    = float("inf")

    while hi - lo > precision:
        mid   = (lo + hi) / 2
        total, _ = _clip_interval_extraction(
            text, video_name, video_clip_dir, mid, fps,
            last, last_last, min_dur, tolerance,
            vtg_model, clip_model, feat_cache_dir, txt_save_dir, device)
        print(f"  binary search: threshold={mid:.4f}  duration={total}")
        if total > text_duration:
            if total < best_dur:
                best_dur, best_thresh = total, mid
            lo = mid
        else:
            hi = mid

    best_dur, best_ivs = _clip_interval_extraction(
        text, video_name, video_clip_dir, best_thresh, fps,
        last, last_last, min_dur, tolerance,
        vtg_model, clip_model, feat_cache_dir, txt_save_dir, device)
    return best_dur, best_ivs


# ── Interview retrieval ───────────────────────────────────────────────────────

def _find_prev_narrator(idx: int, df: pd.DataFrame) -> str:
    for i in range(idx - 1, -1, -1):
        if df.iloc[i]["Tag"] == "Narration":
            return str(df.iloc[i]["Text"]).strip()
    return ""


def _find_next_narrator(idx: int, df: pd.DataFrame) -> str:
    for i in range(idx + 1, len(df)):
        if df.iloc[i]["Tag"] == "Narration":
            return str(df.iloc[i]["Text"]).strip()
    return ""


def call_bart_retriever(prev_narrator: str, next_narrator: str,
                        interview_visual_emb: np.ndarray,
                        interview_txt: Path, interview_json: Path,
                        bart, embedder, bart_tokenizer, fusion_module,
                        device: str):
    """BART infill + FusionModule retrieval (mirrors call_retriver in generate.py)."""
    segments = [line.strip() for line in interview_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(interview_json, encoding="utf-8") as f:
        import json
        interview_data = json.load(f)

    mask_token  = bart_tokenizer.mask_token
    query_ctx   = f"{prev_narrator} {mask_token} {next_narrator}"
    inputs      = bart_tokenizer(query_ctx, return_tensors="pt", padding=True,
                                 truncation=True, max_length=128)
    input_ids   = inputs.input_ids.to(device)
    attn_mask   = inputs.attention_mask.to(device)

    with torch.no_grad():
        bart_out     = bart(input_ids=input_ids, attention_mask=attn_mask,
                            output_hidden_states=True)
        query_emb    = bart_out.decoder_hidden_states[-1][:, -1, :]
        gt_text_emb  = embedder.encode(segments, convert_to_tensor=True).to(device)

    vis_emb   = torch.from_numpy(interview_visual_emb).to(device)
    joint_emb = fusion_module(gt_text_emb, vis_emb)

    query_emb  = F.normalize(query_emb, p=2, dim=-1)
    cosine_sim = F.cosine_similarity(query_emb.unsqueeze(1), joint_emb.unsqueeze(0), dim=-1).squeeze(0)
    cosine_sim = cosine_sim / 0.1  # temperature
    best_idx   = torch.argmax(cosine_sim).item()

    seg = interview_data[best_idx]
    return seg["Start Time"], seg["End Time"], seg.get("part", "main")


# ── Main inference ────────────────────────────────────────────────────────────

def run_inference(args):
    device = args.device

    # ── Load screenplay CSV ───────────────────────────────────────────────────
    screenplay_csv = (
        Path(args.screenplay_dir) / args.video_name[0] / args.video_name / "screenplay_summary.csv"
    )
    if not screenplay_csv.exists():
        print(f"Error: screenplay CSV not found at {screenplay_csv}", flush=True)
        raise FileNotFoundError(screenplay_csv)
    df = pd.read_csv(screenplay_csv)
    print(f"Loaded screenplay: {len(df)} rows")

    # ── Interview base paths ──────────────────────────────────────────────────
    interview_base = Path(args.interview_base) / args.video_name[0] / args.video_name
    interview_txt  = interview_base / "interview.txt"
    interview_json = interview_base / "interview.json"
    interview_npy  = interview_base / "interview_visuals_random.npy"

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading CLIP model...")
    clip_model = clip_module.load("ViT-B/32", device=device, jit=False)[0]

    print("Loading UniVTG model...")
    vtg_model = load_univtg(args.univtg_resume, args.univtg_save_dir)

    print("Loading SentenceTransformer...")
    embedder = SentenceTransformer("all-mpnet-base-v2").to(device)
    embedder.eval()

    print("Loading BART + FusionModule...")
    bart, bart_tokenizer = load_bart(args.bart_model, device)
    fusion_module        = load_fusion(args.fusion_model, device)

    interview_visual_emb = None
    if interview_npy.exists():
        interview_visual_emb = np.load(interview_npy)
        print(f"Loaded visual embeddings: {interview_visual_emb.shape}")

    feat_cache_dir = Path(args.feat_cache_dir)
    feat_cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each row ──────────────────────────────────────────────────────
    time_intervals = []
    last_last: list = []
    last:      list = []

    for idx, row in df.iterrows():
        tag = str(row["Tag"]).strip()
        print(f"\n[{idx+1}/{len(df)}] Tag={tag}  text={str(row['Text'])[:60]}")

        if tag == "Narration":
            text_duration = float(row["Duration"])
            best_dur, best_ivs = binary_search_threshold(
                text        = str(row["Text"]),
                video_name  = args.video_name,
                video_clip_dir = Path(args.video_clip_dir),
                fps         = args.fps,
                text_duration = text_duration,
                last        = last,
                last_last   = last_last,
                min_dur     = args.min_duration,
                tolerance   = args.tolerance,
                vtg_model   = vtg_model,
                clip_model  = clip_model,
                feat_cache_dir = feat_cache_dir,
                txt_save_dir   = args.univtg_save_dir,
                device      = device,
            )
            print(f"  → best_dur={best_dur}  intervals={best_ivs}")
            start_time, end_time = (best_ivs[0][0], best_ivs[-1][1]) if best_ivs else (0, 0)
            part = "main"
            last_last = last
            last      = best_ivs

        else:  # SpeakerID
            prev_narr = _find_prev_narrator(idx, df)
            next_narr = _find_next_narrator(idx, df)
            start_time, end_time, part = call_bart_retriever(
                prev_narr, next_narr, interview_visual_emb,
                interview_txt, interview_json,
                bart, embedder, bart_tokenizer, fusion_module, device,
            )
            print(f"  → start={start_time:.2f}  end={end_time:.2f}  part={part}")

        time_intervals.append({
            "Tag":        tag,
            "Text":       str(row["Text"]),
            "start_time": start_time,
            "end_time":   end_time,
            "part":       part,
        })

    # ── Save output ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir) / args.video_name[0] / args.video_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "time_interval.csv"

    pd.DataFrame(time_intervals).to_csv(out_csv, index=False)
    print(f"\nSaved {len(time_intervals)} rows → {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single-video REGen inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video_name", required=True,
                        help="Video identifier (e.g. 1gBwyk4MGG8)")
    parser.add_argument("--screenplay_dir", required=True,
                        help="Root dir containing <v[0]>/<video_name>/screenplay_summary.csv")
    parser.add_argument("--interview_base", required=True,
                        help="Root dir containing interview.txt / interview.json / interview_visuals_random.npy")
    parser.add_argument("--video_clip_dir", required=True,
                        help="Root dir containing <v[0]>/<video_name>/*.mp4 clips")
    parser.add_argument("--univtg_resume", required=True,
                        help="Path to UniVTG checkpoint (model_best.ckpt)")
    parser.add_argument("--univtg_save_dir", default="/tmp/univtg_feats",
                        help="Temp dir for UniVTG text features (default: /tmp/univtg_feats)")
    parser.add_argument("--feat_cache_dir", default="/tmp/vid_feat_cache",
                        help="Dir for cached CLIP video features (default: /tmp/vid_feat_cache)")
    parser.add_argument("--bart_model", required=True,
                        help="Path to fine-tuned BART .pth")
    parser.add_argument("--fusion_model", required=True,
                        help="Path to FusionModule .pth")
    parser.add_argument("--output_dir", required=True,
                        help="Root output dir; CSV saved under <v[0]>/<video_name>/time_interval.csv")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="Frames per second used during feature extraction (default: 1)")
    parser.add_argument("--min_duration", type=int, default=3,
                        help="Minimum clip duration in frames for interval filtering (default: 3)")
    parser.add_argument("--tolerance", type=int, default=1,
                        help="Overlap tolerance in frames when excluding previous intervals (default: 1)")
    parser.add_argument("--device", default="cuda",
                        help="PyTorch device (default: cuda)")
    args = parser.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()
