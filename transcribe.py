#!/usr/bin/env python3
"""
Voice-to-Text Transcription System with Speaker Diarization (English)
Uses OpenAI Whisper for transcription + pyannote-audio for speaker identification.

Usage:
    # Basic transcription (no speaker labels)
    python transcribe.py audio.mp3

    # Transcription WITH speaker labels
    python transcribe.py video.mp4 --speakers

    # Speaker-labeled SRT subtitles for Premiere Pro
    python transcribe.py video.mp4 --speakers --srt

    # Specify number of speakers (improves accuracy)
    python transcribe.py video.mp4 --speakers --num-speakers 3 --srt

    # Use larger model for better transcription accuracy
    python transcribe.py video.mp4 --speakers --model large-v3 --srt

    # Save to specific file
    python transcribe.py video.mp4 --speakers --srt --output subtitles.srt

    # Plain text with timestamps
    python transcribe.py audio.mp3 --speakers --timestamps

Setup:
    pip install openai-whisper torch pyannote.audio sounddevice soundfile numpy

    Speaker diarization requires a free Hugging Face token:
    1. Create account at https://huggingface.co
    2. Accept the model terms at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
    3. Get your token at: https://huggingface.co/settings/tokens
    4. Run: huggingface-cli login
       OR pass --hf-token YOUR_TOKEN
"""

import argparse
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def get_device():
    """
    Detect the best available compute device.

    Priority: CUDA (NVIDIA) > DirectML (AMD/Intel on Windows) > CPU
    DirectML requires: pip install torch-directml
    """
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")

    try:
        import torch_directml
        dml = torch_directml.device()
        return "directml", dml
    except ImportError:
        pass

    return "cpu", torch.device("cpu")


def check_dependencies(need_speakers=False):
    """Check and report missing dependencies."""
    missing = []
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("openai-whisper")
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    if need_speakers:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            missing.append("pyannote.audio")

    if missing:
        print("❌ Missing required packages:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)


def check_recording_dependencies():
    """Check dependencies needed for microphone recording."""
    missing = []
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")

    if missing:
        print("❌ Missing recording packages:")
        print(f"   pip install {' '.join(missing)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Audio recording
# ---------------------------------------------------------------------------

def record_audio(duration: int, sample_rate: int = 16000) -> str:
    """Record audio from the microphone and save to a temp file."""
    import sounddevice as sd
    import soundfile as sf
    import numpy as np

    temp_path = "recorded_audio.wav"

    print(f"\n🎙️  Recording for {duration} seconds... (Speak now!)")
    print("   Press Ctrl+C to stop early.\n")

    try:
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        for i in range(duration, 0, -1):
            print(f"   ⏱️  {i}s remaining...", end="\r")
            time.sleep(1)
        sd.wait()
        print("   ✅ Recording complete!       ")
    except KeyboardInterrupt:
        sd.stop()
        print("\n   ⏹️  Recording stopped early.")
        audio_data = audio_data[: len(audio_data)]

    sf.write(temp_path, audio_data, sample_rate)
    return temp_path


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: str = "en",
    timestamps: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path:  Path to audio file (mp3, wav, m4a, flac, mp4, etc.)
        model_name:  tiny | base | small | medium | large-v3
        language:    Language code (default: "en")
        timestamps:  Include word-level timestamps
        verbose:     Print progress

    Returns:
        dict with 'text' and 'segments'
    """
    import whisper
    import torch

    device_name, device = get_device()
    if device_name == "cuda":
        print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif device_name == "directml":
        print("🚀 Using AMD/Intel GPU via DirectML")
    else:
        print("💻 Using CPU (transcription will be slower)")
        if model_name in ("large-v3", "large", "medium"):
            print(f"   ⚠️  '{model_name}' is slow on CPU. Consider 'base' or 'small'.")

    print(f"📦 Loading Whisper model: {model_name}...")
    # Load on CPU first, then move to device (required for DirectML)
    model = whisper.load_model(model_name, device="cpu")
    if device_name != "cpu":
        model = model.to(device)

    print(f"🔊 Transcribing: {audio_path}...")
    t0 = time.time()

    result = model.transcribe(
        audio_path,
        language=language,
        verbose=verbose,
        word_timestamps=timestamps,
        fp16=(device_name == "cuda"),  # fp16 only supported on CUDA
    )

    print(f"✅ Transcription done in {time.time() - t0:.1f}s\n")
    return result


# ---------------------------------------------------------------------------
# Speaker diarization (pyannote)
# ---------------------------------------------------------------------------

def diarize(audio_path: str, hf_token: str = None, num_speakers: int = None) -> list:
    """
    Run speaker diarization on an audio file using pyannote.

    Args:
        audio_path:    Path to audio file
        hf_token:      Hugging Face token (or uses cached login)
        num_speakers:  Exact number of speakers (None = auto-detect)

    Returns:
        List of dicts: [{"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"}, ...]
    """
    from pyannote.audio import Pipeline

    print("🗣️  Loading speaker diarization model...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    except Exception as e:
        error_msg = str(e)
        if "token" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print("\n❌ Hugging Face authentication failed.")
            print("   To use speaker diarization:")
            print("   1. Create account: https://huggingface.co")
            print("   2. Accept model terms:")
            print("      https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("      https://huggingface.co/pyannote/segmentation-3.0")
            print("   3. Run: huggingface-cli login")
            print("      OR pass: --hf-token YOUR_TOKEN")
            sys.exit(1)
        raise

    # pyannote only supports CUDA or CPU (not DirectML)
    import torch
    if torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
    else:
        print("   ℹ️  Speaker diarization running on CPU (DirectML not supported by pyannote)")

    print(f"🔍 Identifying speakers in: {audio_path}...")
    t0 = time.time()

    diarization_params = {}
    if num_speakers is not None:
        diarization_params["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **diarization_params)

    # Convert to simple list of segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Count unique speakers
    unique_speakers = set(s["speaker"] for s in segments)
    elapsed = time.time() - t0
    print(f"✅ Found {len(unique_speakers)} speaker(s) in {elapsed:.1f}s\n")

    return segments


def assign_speakers(whisper_result: dict, diarization_segments: list) -> dict:
    """
    Merge Whisper transcription with pyannote speaker labels.

    For each Whisper segment, finds which speaker was talking during
    the majority of that segment's time span.

    Args:
        whisper_result:        Whisper transcription result
        diarization_segments:  Speaker diarization segments

    Returns:
        Updated whisper_result with 'speaker' field added to each segment
    """
    for seg in whisper_result["segments"]:
        seg_start = seg["start"]
        seg_end = seg["end"]

        # Calculate overlap of each speaker with this segment
        speaker_overlap = {}
        for d_seg in diarization_segments:
            overlap_start = max(seg_start, d_seg["start"])
            overlap_end = min(seg_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = d_seg["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

        # Assign the speaker with the most overlap
        if speaker_overlap:
            seg["speaker"] = max(speaker_overlap, key=speaker_overlap.get)
        else:
            seg["speaker"] = "UNKNOWN"

    # Rename speakers to friendly names (Speaker 1, Speaker 2, ...)
    seen_speakers = {}
    counter = 1
    for seg in whisper_result["segments"]:
        raw = seg["speaker"]
        if raw not in seen_speakers:
            seen_speakers[raw] = f"Speaker {counter}"
            counter += 1
        seg["speaker"] = seen_speakers[raw]

    return whisper_result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_timestamp(seconds: float, srt_format: bool = False) -> str:
    """Convert seconds to timestamp format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    if srt_format:
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{ms:03d}"


def format_srt(result: dict, max_chars: int = 42, has_speakers: bool = False) -> str:
    """Format transcription as SRT subtitles for Premiere Pro.

    Example with speakers:
        1
        00:00:01,000 --> 00:00:03,500
        [Speaker 1] Hello everyone, welcome
        to today's meeting.
    """
    srt_blocks = []
    index = 1

    for seg in result["segments"]:
        start = format_timestamp(seg["start"], srt_format=True)
        end = format_timestamp(seg["end"], srt_format=True)
        text = seg["text"].strip()

        # Prepend speaker label
        if has_speakers and "speaker" in seg:
            text = f"[{seg['speaker']}] {text}"

        # Wrap long lines into max 2 lines for readability
        if len(text) > max_chars:
            words = text.split()
            line1 = ""
            line2 = ""
            for word in words:
                if len(line1) + len(word) + 1 <= max_chars:
                    line1 = f"{line1} {word}".strip()
                else:
                    line2 = f"{line2} {word}".strip()
            text = f"{line1}\n{line2}" if line2 else line1

        srt_blocks.append(f"{index}\n{start} --> {end}\n{text}\n")
        index += 1

    return "\n".join(srt_blocks)


def format_output(result: dict, show_timestamps: bool = False, has_speakers: bool = False) -> str:
    """Format transcription as readable text."""
    if not show_timestamps and not has_speakers:
        return result["text"].strip()

    lines = []
    for seg in result["segments"]:
        parts = []

        if show_timestamps:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            parts.append(f"[{start} --> {end}]")

        if has_speakers and "speaker" in seg:
            parts.append(f"[{seg['speaker']}]")

        parts.append(seg["text"].strip())
        lines.append("  ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🎤 Speech-to-Text with Speaker Diarization (Whisper + pyannote)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Models (accuracy vs speed):
  tiny      Fastest, least accurate     (~39M params)
  base      Good default balance        (~74M params)
  small     Noticeably better accuracy  (~244M params)
  medium    High accuracy               (~769M params)
  large-v3  Best available accuracy     (~1550M params)

Examples:
  python transcribe.py meeting.mp3
  python transcribe.py video.mp4 --speakers --srt
  python transcribe.py video.mp4 --speakers --num-speakers 3 --model large-v3 --srt
  python transcribe.py interview.wav --speakers --timestamps -o transcript.txt
  python transcribe.py --record --duration 60 --speakers --srt
        """,
    )
    parser.add_argument("audio", nargs="?", help="Path to audio/video file")
    parser.add_argument(
        "--model", "-m", default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--output", "-o", help="Save transcript to file",
    )
    parser.add_argument(
        "--timestamps", "-t", action="store_true",
        help="Include timestamps in output",
    )
    parser.add_argument(
        "--srt", action="store_true",
        help="Output as SRT subtitle file (for Premiere Pro, DaVinci, etc.)",
    )
    parser.add_argument(
        "--speakers", action="store_true",
        help="Enable speaker diarization (identify who is speaking)",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Exact number of speakers (improves diarization accuracy)",
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="Hugging Face token for pyannote (or use huggingface-cli login)",
    )
    parser.add_argument(
        "--record", "-r", action="store_true",
        help="Record from microphone",
    )
    parser.add_argument(
        "--duration", "-d", type=int, default=30,
        help="Recording duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress",
    )

    args = parser.parse_args()

    if not args.record and not args.audio:
        parser.error("Provide an audio file path, or use --record to record from mic.")

    check_dependencies(need_speakers=args.speakers)

    # ---- Get audio source ----
    if args.record:
        check_recording_dependencies()
        audio_path = record_audio(args.duration)
    else:
        audio_path = args.audio
        if not os.path.isfile(audio_path):
            print(f"❌ File not found: {audio_path}")
            sys.exit(1)

    # ---- Transcribe with Whisper ----
    result = transcribe(
        audio_path=audio_path,
        model_name=args.model,
        timestamps=args.timestamps or args.srt,
        verbose=args.verbose,
    )

    # ---- Speaker diarization ----
    if args.speakers:
        diarization_segments = diarize(
            audio_path=audio_path,
            hf_token=args.hf_token,
            num_speakers=args.num_speakers,
        )
        result = assign_speakers(result, diarization_segments)

    # ---- Format output ----
    use_srt = args.srt or (args.output and args.output.lower().endswith(".srt"))

    if use_srt:
        output = format_srt(result, has_speakers=args.speakers)
        print("=" * 60)
        print("📝 SRT SUBTITLES")
        print("=" * 60)
        print(output)
        print("=" * 60)

        if not args.output:
            base = os.path.splitext(audio_path)[0]
            args.output = f"{base}.srt"
            print(f"\n💾 Auto-saving SRT to: {args.output}")
    else:
        output = format_output(result, show_timestamps=args.timestamps, has_speakers=args.speakers)
        print("=" * 60)
        print("📝 TRANSCRIPT")
        print("=" * 60)
        print(output)
        print("=" * 60)

    # ---- Save ----
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"\n💾 Saved to: {args.output}")

    # ---- Cleanup ----
    if args.record and os.path.exists("recorded_audio.wav"):
        os.remove("recorded_audio.wav")


if __name__ == "__main__":
    main()
