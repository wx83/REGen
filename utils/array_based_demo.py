from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import sys
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    AudioClip,
    concatenate_videoclips,
    concatenate_audioclips
)
from moviepy.video.VideoClip import ColorClip
from moviepy.audio.AudioClip import AudioArrayClip

def build_compilation(
    csv_file: Path,
    video_file: Path,
    narration_audio_dir: Path,
    output_file: Path
):
    df = pd.read_csv(csv_file)
    video = VideoFileClip(str(video_file))
    total_dur = video.duration
    print(f"total duration is = {total_dur}")
    narration_audio_cache = {}
    final_clips = []

    i, n = 0, len(df)
    while i < n:
        row = df.iloc[i]
        tag = str(row["tag"]).strip().lower()

        # ----- narration segments -----
        if tag == "narration":
            nid = int(row["narration_id"])
            times = []
            while (
                i < n
                and str(df.iloc[i]["tag"]).strip().lower() == "narration"
                and int(df.iloc[i]["narration_id"]) == nid
            ):
                s = float(df.iloc[i]["start_time"])
                e = min(float(df.iloc[i]["end_time"]), total_dur)
                times.append((s, e))
                i += 1

            # build the visual subclip
            subclips = [video.subclip(s, e) for s, e in times]
            narr_video = concatenate_videoclips(subclips, method="compose")
            vid_dur = narr_video.duration
            w, h = narr_video.size
            vfps = narr_video.fps

            # load & cache audio
            if nid not in narration_audio_cache:
                path = narration_audio_dir / f"chunk_{nid}_speech.wav"
                narration_audio_cache[nid] = AudioFileClip(str(path))
            audio = narration_audio_cache[nid]
            aud_dur = audio.duration
            afps = audio.fps

            # CASE 1: audio shorter → pad audio with silence to match video
            if aud_dur < vid_dur:
                n_silence = int((vid_dur - aud_dur) * afps)
                nch = audio.to_soundarray(fps=afps).shape[1]
                silence = np.zeros((n_silence, nch))
                padded_arr = np.vstack([audio.to_soundarray(fps=afps), silence])
                final_audio = AudioArrayClip(padded_arr, fps=afps)
                final_video = narr_video

            # CASE 2: audio longer → pad video with black frames
            elif aud_dur > vid_dur:
                black = ColorClip(
                    size=(w, h),
                    color=(0, 0, 0),
                    duration=(aud_dur - vid_dur)
                ).set_fps(vfps)
                final_video = concatenate_videoclips(
                    [narr_video, black], method="compose"
                )
                final_audio = audio

            # CASE 3: equal → no padding
            else:
                final_video, final_audio = narr_video, audio

            # attach and collect
            final_clips.append(final_video.set_audio(final_audio))

        # ----- non-narration: use original audio/video -----
        else:
            start = float(row["start_time"])
            end = min(float(row["end_time"]), total_dur)

            clip = video.subclip(start, end)
            if clip.audio is None:
                print(f"Warning: no audio for clip {start}-{end}")
            final_clips.append(clip)
            i += 1

    # stitch entire sequence & export
    final = concatenate_videoclips(final_clips, method="compose")
    final.write_videofile(
        str(output_file),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )

def generate_scale(video_name_list, audio_input, input_dir, output_dir, video_dir, output_name,csv_name):
    for video_name in video_name_list:
        csv_path = input_dir / video_name[0] / video_name / f"{csv_name}.csv"
        video_path = video_dir / video_name[0] / video_name / "main.mp4"
        narration_audio_path = audio_input / video_name[0] / video_name
        output_path = output_dir / video_name / f"{output_name}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_compilation(
            csv_file=csv_path,
            video_file=video_path,
            narration_audio_dir=narration_audio_path,
            output_file=output_path
        )
        print(f"Processed {video_name} to {output_path}")

def generate_scale_lecture(video_name_list, audio_input, input_dir, output_dir, video_dir, output_name,csv_name):
    for video_name in video_name_list:
        csv_path = input_dir / video_name[0] / video_name / f"{csv_name}.csv"
        video_path = video_dir / f"{video_name}.mp4"
        narration_audio_path = audio_input / video_name[0] / video_name
        output_path = output_dir / video_name / f"{output_name}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        build_compilation(
            csv_file=csv_path,
            video_file=video_path,
            narration_audio_dir=narration_audio_path,
            output_file=output_path
        )
        print(f"Processed {video_name} to {output_path}")

if __name__ == "__main__":
    pass