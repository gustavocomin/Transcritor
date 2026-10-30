<p align="center">
  <strong>🌐 Language / Idioma:</strong>
  <a href="README.md">🇧🇷 Português</a> | 🇺🇸 <strong>English</strong>
</p>

# Transcritor

Audio and video transcription in Portuguese using [Whisper](https://github.com/SYSTRAN/faster-whisper), running **locally** on your GPU. The application extracts audio from video, transcribes it, identifies speakers, and generates an AI-powered PDF summary, all through a simple Windows interface.

## Features

The application has three tabs:

1. **Extract audio from video**: converts a video (`.mp4`, `.mkv`, `.mov`, `.webm`, etc.) to a 16 kHz mono `.wav` file, Whisper's native audio format. Optionally normalizes volume and filters out low frequencies.
2. **Transcribe audio**: transcribes audio with Whisper (*small*, *medium*, or *large-v3* models) and generates:
   - a `.txt` file with timestamps for each segment;
   - an `.srt` subtitle file with start and end times;
   - **speaker diarization** (optional): identifies "Person 1", "Person 2", etc., and opens a window where you can assign names;
   - **AI review** (optional): a PDF with a title, topic-based summaries, action items, text organized chronologically, and a list of uncertain segments.
3. **Review and PDF**: works with existing `.txt`, `.srt`, or reviewed `.md` files without transcribing again. Assign speaker names, rerun the AI review, and generate the PDF.

All output files are saved in a folder named after the original file:

```
reuniao.mkv
reuniao\
    reuniao.wav
    reuniao.txt
    reuniao.srt
    reuniao - revisado.pdf
```

## Requirements

| | |
|---|---|
| Operating system | Windows 10/11 (uses Edge or Chrome to generate PDFs) |
| Python | **3.10 to 3.13** (tested with 3.13) |
| Graphics card | NVIDIA recommended (tested on a 6 GB RTX 2060). CPU execution is supported but significantly slower |
| Disk space | Approximately 10 GB free (PyTorch with CUDA + models) |

The installation script handles the remaining dependencies (ffmpeg, PyTorch, Whisper, and pyannote).

## Installation

```powershell
git clone https://github.com/gustavocomin/Transcritor.git
cd Transcritor
powershell -ExecutionPolicy Bypass -File .\instalar.ps1
```

The script:
1. checks your Python installation;
2. installs **ffmpeg** through `winget` if needed;
3. creates a `.venv` virtual environment in the project directory;
4. installs **PyTorch** (with CUDA if an NVIDIA GPU is available), followed by the packages in `requirements.txt`;
5. creates a **Transcritor** shortcut in the Start Menu.

Options: `-Cpu` installs CPU-only PyTorch, and `-SemAtalho` skips shortcut creation.

To launch the application, search for "Transcritor" in the Start Menu or double-click `Transcritor.bat`.

Each model is downloaded automatically the first time you use it (*large-v3* is approximately 3 GB). Models are stored in your user's Hugging Face cache (`%USERPROFILE%\.cache\huggingface`).

**Update:** run `git pull`, then run `instalar.ps1` again (it reuses previously installed components).

**Uninstall:** delete the project directory and the `Transcritor` Start Menu shortcut. Optionally, remove the `HF_TOKEN` environment variable and cached models from `%USERPROFILE%\.cache\huggingface`.

## Choosing a model

| UI option | Model | When to use |
|---|---|---|
| Rápido (Fast) | `small` | Testing and clear recordings; lightest option |
| Médio (Medium) | `medium` | Balance between speed and accuracy |
| Alta qualidade (High quality) | `large-v3` | Noisy recordings, distant voices, or multiple speakers; the default |

The *large-v3* model fits on a 6 GB GPU using `float16`. If GPU memory is insufficient, the application tries `int8_float16` and, as a last resort, the CPU. If the GPU fails during transcription, the application reruns the entire transcription on the CPU and displays a notification.

## Speaker diarization (Hugging Face)

Speaker diarization uses the [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) model, which requires a free account:

1. Create an account at <https://huggingface.co/join>.
2. Open the model page, complete the form (the "Company/university" and "Website" fields can use, for example, "Independent" and your profile), and click **Agree and access repository**.
3. Create a **Read** token at <https://huggingface.co/settings/tokens>.
4. In the application, check **Separar por quem fala** (Separate by speaker). On first use, the application requests your token and stores it in your user-level `HF_TOKEN` environment variable. You will not need to enter it again.

Specify the number of speakers when known (this helps the model); `0` enables automatic detection. Overlapping speech, similar voices, and speakers far from the microphone are common sources of errors, so treat speaker labels as approximate.

## AI review

The AI review attempts the following, in order:
1. **Claude Code** (`claude -p`);
2. if that fails (usage limit or other error), **Codex** (`codex exec`);
3. if both fail, the application notifies you. Your transcription (`.txt` and `.srt`) is already saved, and you can generate the review later from tab 3.

Claude Code and/or Codex must be installed and authenticated. Otherwise, uncheck **Gerar revisão com IA** (Generate AI review); all other features will continue to work.

**Privacy:** the transcript text is sent to the service powering the selected AI reviewer. Disable AI review if the content is sensitive.

The prompt sent to the AI is stored in [`prompt_revisao.txt`](prompt_revisao.txt) and can be edited as needed (for instance, to request a different summary format).

## Troubleshooting

- **Logs:** errors and processing steps are recorded in `transcritor.log`, in the application directory.
- **`Library cublas64_12.dll is not found`:** CUDA-enabled PyTorch was not installed. Run `instalar.ps1` again and verify that you have an NVIDIA GPU with up-to-date drivers.
- **PDF is not generated:** Edge or Chrome must be installed. If PDF generation fails, the review is saved as an `.md` file.
- **Speaker diarization returns a 401 error:** accept the `community-1` model terms using the same account associated with your token.

## Project structure

| File | Purpose |
|---|---|
| `transcritor.py` | Main application (UI, transcription, speakers, AI review, and PDF generation) |
| `prompt_revisao.txt` | Instructions sent to the AI reviewer |
| `instalar.ps1` | Dependency and shortcut installation |
| `requirements.txt` | Python dependencies (PyTorch is installed first by the script) |
| `Transcritor.bat` | Starts the application without installing a shortcut |
| `LICENSE` | MIT license |

## License

Distributed under the [MIT license](LICENSE). The application neither bundles nor redistributes AI models or ffmpeg: users obtain them during installation, and the models have their own licenses. The `pyannote/speaker-diarization-community-1` speaker diarization model is licensed under **CC-BY-4.0** and requires attribution to the [pyannote](https://github.com/pyannote/pyannote-audio) authors.
