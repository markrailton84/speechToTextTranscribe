# transcribe.py

A command-line speech-to-text tool that transcribes audio/video files using **OpenAI Whisper**, with optional **speaker diarization** (who said what) via **pyannote-audio**. Outputs plain text, timestamped transcripts, or SRT subtitle files.

---

## Features

- Transcribe any audio or video file (mp3, mp4, wav, m4a, flac, etc.)
- **Speaker diarization** — identify and label individual speakers
- **SRT subtitle output** — ready for Premiere Pro, DaVinci Resolve, etc.
- **Microphone recording** — record directly from your mic and transcribe live
- Five Whisper model sizes to balance speed vs accuracy
- GPU acceleration — NVIDIA (CUDA), AMD/Intel on Windows (DirectML), or CPU fallback

---

## Requirements

### NVIDIA GPU (CUDA)
```bash
pip install openai-whisper torch pyannote.audio sounddevice soundfile numpy
```

### AMD / Intel GPU on Windows (DirectML)
```bash
pip install openai-whisper torch pyannote.audio sounddevice soundfile numpy torch-directml
```
The script auto-detects DirectML when `torch-directml` is installed. No other changes needed.
> **Note:** Speaker diarization (`--speakers`) always runs on CPU when using DirectML — pyannote does not support it.

### CPU only
```bash
pip install openai-whisper torch pyannote.audio sounddevice soundfile numpy
```

### Speaker Diarization Setup (first time only)

Speaker labelling requires a free Hugging Face account and model access:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Accept the model terms at:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Login via CLI:
   ```bash
   huggingface-cli login
   ```
   Or pass your token directly with `--hf-token YOUR_TOKEN`

---

## Usage

```bash
# Basic transcription
python transcribe.py audio.mp3

# With speaker labels
python transcribe.py meeting.mp4 --speakers

# SRT subtitles with speaker labels (auto-saved as meeting.srt)
python transcribe.py meeting.mp4 --speakers --srt

# Specify exact number of speakers (improves accuracy)
python transcribe.py interview.mp4 --speakers --num-speakers 2 --srt

# Use a larger model for better accuracy
python transcribe.py meeting.mp4 --speakers --model large-v3 --srt

# Save transcript to a file
python transcribe.py audio.mp3 --speakers --timestamps -o transcript.txt

# Record from microphone (30 seconds)
python transcribe.py --record

# Record for 60 seconds with speaker labels
python transcribe.py --record --duration 60 --speakers --srt
```

---

## Arguments

| Argument | Short | Description |
|---|---|---|
| `audio` | | Path to audio/video file |
| `--model` | `-m` | Whisper model size (default: `base`) |
| `--output` | `-o` | Save transcript to file |
| `--timestamps` | `-t` | Include timestamps in output |
| `--srt` | | Output as SRT subtitle file |
| `--speakers` | | Enable speaker diarization |
| `--num-speakers` | | Number of speakers (improves accuracy) |
| `--hf-token` | | Hugging Face token for pyannote |
| `--record` | `-r` | Record from microphone |
| `--duration` | `-d` | Recording duration in seconds (default: 30) |
| `--verbose` | `-v` | Show detailed progress |

---

## Whisper Models

| Model | Parameters | Notes |
|---|---|---|
| `tiny` | 39M | Fastest, least accurate |
| `base` | 74M | Good default |
| `small` | 244M | Noticeably better accuracy |
| `medium` | 769M | High accuracy |
| `large-v3` | 1550M | Best available (slow on CPU) |

> **Tip:** On CPU, stick to `base` or `small`. Use `large-v3` if you have a GPU.

---

## Running in Docker (CPU)

Docker runs CPU-only (AMD GPU passthrough is not supported on Windows Docker).

### Setup

```bash
# Create a folder for your audio files
mkdir audio

# Copy your files in
cp meeting.mp4 audio/
```

### Build & Run

```bash
docker compose build

# Basic transcription
docker compose run transcribe /audio/meeting.mp3

# With speaker labels and SRT output
docker compose run transcribe /audio/meeting.mp4 --speakers --srt

# With Hugging Face token for diarization
docker compose run transcribe /audio/meeting.mp4 --speakers --hf-token YOUR_TOKEN --srt
```

Output files (`.srt`, `.txt`) are saved inside `/audio/` in the container, which maps back to your local `./audio/` folder.

> **Note:** The `--record` (microphone) flag does not work inside Docker.

---

## Output Examples

**Plain text:**
```
Hello everyone, welcome to today's meeting. Let's get started.
```

**With timestamps and speakers:**
```
[00:00:01.000 --> 00:00:03.500]  [Speaker 1]  Hello everyone, welcome to today's meeting.
[00:00:04.000 --> 00:00:06.200]  [Speaker 2]  Thanks for joining. Let's get started.
```

**SRT format:**
```
1
00:00:01,000 --> 00:00:03,500
[Speaker 1] Hello everyone, welcome
to today's meeting.

2
00:00:04,000 --> 00:00:06,200
[Speaker 2] Thanks for joining.
Let's get started.
```
