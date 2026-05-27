import os
import re
import sys
import uuid
import json
import string
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory

import librosa
import numpy as np

app = Flask(__name__, static_folder="static")

# Windows-safe work directory
WORK_DIR = Path(os.getenv("TEMP", "/tmp")) / "audiolab"
WORK_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOADS_DIR = Path.home() / "Downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
LOW_RESOURCE_MODE = os.getenv("LOW_RESOURCE_MODE", "0").strip() not in ("0", "false", "False")
MAX_STEM_SECONDS = int(os.getenv("MAX_STEM_SECONDS", "600"))
DEMUCS_DEVICE         = os.getenv("DEMUCS_DEVICE", "cuda").strip().lower()
FREE_TIER             = os.getenv("FREE_TIER", "1").strip() not in ("0", "false", "False")
_dev_tier_override: "bool | None" = None  # toggled at runtime for dev testing

def is_free_tier() -> bool:
    return _dev_tier_override if _dev_tier_override is not None else FREE_TIER
STRIPE_SECRET         = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_ID       = os.getenv("STRIPE_PRICE_ID", "")
APP_DOWNLOAD_PATH     = os.getenv("APP_DOWNLOAD_PATH", "")
# Job store
jobs = {}

# One-time download tokens issued after Stripe payment
_download_tokens: dict = {}  # token -> {expires, path, uses_left}
_token_lock = threading.Lock()



NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot harmonic mixing wheel — indexed by semitone (C=0, C#=1, … B=11)
CAMELOT_MAJOR = ["5B","12B","7B","2B","9B","4B","11B","6B","1B","8B","3B","10B"]
CAMELOT_MINOR = ["2A","9A","4A","11A","6A","1A","8A","3A","10A","5A","12A","7A"]

def default_venv_python(venv_folder_name):
    """Resolve main interpreter path for a project-local venv (Windows vs Unix)."""
    project_dir = Path(__file__).resolve().parent
    if sys.platform == "win32":
        return project_dir / venv_folder_name / "Scripts" / "python.exe"
    return project_dir / venv_folder_name / "bin" / "python"



def resolve_demucs_device():
    """
    Resolve separation device without reducing model quality.
    Supported env values: auto, cpu, cuda, mps, rocm.
    """
    valid = {"auto", "cpu", "cuda", "mps", "rocm"}
    requested = DEMUCS_DEVICE if DEMUCS_DEVICE in valid else "auto"

    # cpu/mps explicit: return immediately (no torch needed).
    if requested == "cpu":
        return "cpu"

    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"

    # Determine which device to probe first based on the request.
    want_cuda = requested in ("cuda", "rocm", "auto")
    want_mps  = requested in ("mps", "auto")

    if want_cuda and torch.cuda.is_available():
        try:
            # Allocating a real tensor forces _lazy_init and confirms the build
            # has the CUDA runtime compiled in, not just the driver headers.
            torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            pass  # CPU-only build or broken driver — fall through

    if want_mps:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            try:
                torch.zeros(1, device="mps")
                return "mps"
            except Exception:
                pass

    return "cpu"


def build_cuda_alloc_conf():
    """
    Derive PYTORCH_CUDA_ALLOC_CONF from live GPU properties via nvidia-smi.

    Uses nvidia-smi rather than torch.cuda so no CUDA context needs to be
    initialised in the server process, avoiding the 'not compiled with CUDA'
    error that occurs when querying torch outside the Demucs subprocess.

    Returns (conf_str, description) so the caller can log what was chosen.
    """
    try:
        smi = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
        result = subprocess.run(
            [smi,
             "--query-gpu=name,compute_cap,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nvidia-smi non-zero exit")

        row = result.stdout.strip().splitlines()[0]
        name, compute_str, free_str, total_str = [p.strip() for p in row.split(",")]
        free_mb  = int(free_str)
        total_mb = int(total_str)
        major, minor = compute_str.split(".")
        compute = int(major) * 10 + int(minor)  # e.g. "8.6" -> 86

        # Aggressiveness scales with architecture: newer controllers
        # reclaim/reallocate faster and carry less driver/display overhead.
        if compute >= 120:   # Blackwell - RTX 50xx, B100
            fraction = 0.92
        elif compute >= 89:  # Ada Lovelace - RTX 40xx, L40
            fraction = 0.88
        elif compute >= 80:  # Ampere - RTX 30xx, A100, A10
            fraction = 0.82
        elif compute >= 75:  # Turing - RTX 20xx, T4
            fraction = 0.75
        elif compute >= 70:  # Volta - V100
            fraction = 0.68
        else:                # Pascal sm_60/61 and older
            fraction = 0.60

        max_split_mb = max(128, min(int(free_mb * fraction), total_mb))

        conf_parts = [
            f"max_split_size_mb:{max_split_mb}",
            "expandable_segments:True",
            "garbage_collection_threshold:0.8",
        ]
        if compute >= 80:
            conf_parts.append("roundup_power2_divisions:16")

        desc = (
            f"{name} sm_{major}{minor} | "
            f"{free_mb} MB free / {total_mb} MB total | "
            f"fraction={fraction:.0%} -> max_split={max_split_mb} MB"
        )
        return ",".join(conf_parts), desc
    except Exception as exc:
        return (
            "expandable_segments:True,garbage_collection_threshold:0.8",
            f"GPU query failed ({exc}), minimal config",
        )


CHAINS_DIR     = Path.home() / "Documents" / "URL2AUDIO" / "VocalChains"
CHAINS_ASSET   = Path(__file__).parent / "assets" / "chains"
CHAIN_PRICE    = float(os.getenv("CHAIN_PRICE", "10.00"))
CHAIN_PRICE_ID = os.getenv("CHAIN_PRICE_ID", "")
CHAINS_DIR.mkdir(parents=True, exist_ok=True)

_CHAIN_COLORS = ["#753BBD", "#9d4edd", "#e040fb", "#ff6b6b", "#ff9a00", "#06b6d4", "#22c55e"]


def load_vocal_chains():
    """Build the chain catalog dynamically from assets/chains/*.fst."""
    if not CHAINS_ASSET.is_dir():
        return []
    chains = []
    for i, fst in enumerate(sorted(CHAINS_ASSET.glob("*.fst"))):
        safe_id = re.sub(r"[^a-z0-9]", "_", fst.stem.lower()).strip("_")
        is_free = fst.stem.lower() == "record"
        chains.append({
            "id":       safe_id,
            "name":     fst.stem,
            "file":     fst.name,
            "price":    0.0 if is_free else CHAIN_PRICE,
            "price_id": "" if is_free else CHAIN_PRICE_ID,
            "free":     is_free,
            "color":    _CHAIN_COLORS[i % len(_CHAIN_COLORS)],
        })
    return chains


VOCAL_CHAINS = load_vocal_chains()


def find_fl_studio() -> "str | None":
    """Locate FL Studio on Windows, macOS, or Linux.

    Returns the executable path (Windows/Linux native), the .app bundle path
    (macOS — use launch_fl_studio() which calls `open -a`), or None if not found.
    """
    if sys.platform == "win32":
        try:
            import winreg
            for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
                for subkey in (
                    r"SOFTWARE\Image-Line\FL Studio",
                    r"SOFTWARE\WOW6432Node\Image-Line\FL Studio",
                ):
                    try:
                        key = winreg.OpenKey(hive, subkey)
                        install_dir, _ = winreg.QueryValueEx(key, "Install_Dir")
                        for exe in ("FL64.exe", "FL.exe"):
                            p = Path(install_dir) / exe
                            if p.is_file():
                                return str(p)
                    except OSError:
                        pass
        except ImportError:
            pass
        bases = [
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Image-Line",
            Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Image-Line",
        ]
        for base in bases:
            for ver in ("FL Studio 2025", "FL Studio 2024", "FL Studio 21", "FL Studio 20", "FL Studio"):
                for exe in ("FL64.exe", "FL.exe"):
                    p = base / ver / exe
                    if p.is_file():
                        return str(p)
        return None

    elif sys.platform == "darwin":
        candidates = [
            Path("/Applications/FL Studio.app"),
            Path(os.path.expanduser("~/Applications/FL Studio.app")),
        ]
        for app in candidates:
            if app.is_dir():
                return str(app)
        # Spotlight fallback
        try:
            result = subprocess.run(
                ["mdfind", "kMDItemCFBundleIdentifier == 'com.image-line.flstudio'"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().splitlines():
                p = Path(line.strip())
                if p.exists():
                    return str(p)
        except Exception:
            pass
        return None

    else:  # Linux
        # Native binary on PATH or common install locations
        for name in ("flstudio", "fl-studio"):
            found = shutil.which(name)
            if found:
                return found
        native = [
            Path("/opt/flstudio/flstudio"),
            Path("/opt/fl-studio/fl-studio"),
            Path(os.path.expanduser("~/.local/bin/flstudio")),
            Path(os.path.expanduser("~/Applications/flstudio.AppImage")),
            Path(os.path.expanduser("~/Applications/FL Studio.AppImage")),
        ]
        for p in native:
            if p.is_file():
                return str(p)
        # Wine-based installation
        for wine_base in (
            Path(os.path.expanduser("~/.wine/drive_c/Program Files/Image-Line")),
            Path(os.path.expanduser("~/.wine/drive_c/Program Files (x86)/Image-Line")),
        ):
            for ver in ("FL Studio 2025", "FL Studio 2024", "FL Studio 21", "FL Studio 20", "FL Studio"):
                for exe in ("FL64.exe", "FL.exe"):
                    p = wine_base / ver / exe
                    if p.is_file():
                        return str(p)
        return None


def launch_fl_studio(fl_path: str, file_path: Path) -> None:
    """Open a .flp file in FL Studio, handling platform differences."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "FL Studio", str(file_path)], close_fds=True)
    elif sys.platform != "win32" and fl_path.endswith(".exe"):
        # Linux + Wine executable
        subprocess.Popen(["wine", fl_path, str(file_path)], close_fds=True)
    else:
        subprocess.Popen([fl_path, str(file_path)], close_fds=True)


def find_tool(name):
    """Find yt-dlp or ffmpeg on PATH, next to server.py, or common Windows locations."""
    found = shutil.which(name)
    if found:
        return found
    script_dir = Path(__file__).parent
    for ext in ("", ".exe"):
        candidate = script_dir / (name + ext)
        if candidate.is_file():
            return str(candidate)
    win_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / name / (name + ".exe"),
        Path("C:/ffmpeg/bin") / (name + ".exe"),
        Path("C:/yt-dlp") / (name + ".exe"),
        Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims" / (name + ".exe"),
    ]
    for p in win_paths:
        if p.is_file():
            return str(p)
    return None


def run(cmd_list, **kwargs):
    tool = cmd_list[0]
    resolved = find_tool(tool)
    if not resolved:
        raise RuntimeError(
            f"'{tool}' not found. Install it and ensure it is on your PATH, "
            f"or place {tool}.exe next to server.py.\n"
            f"  yt-dlp:  https://github.com/yt-dlp/yt-dlp/releases\n"
            f"  ffmpeg:  https://ffmpeg.org/download.html"
        )
    return subprocess.run([resolved] + cmd_list[1:], **kwargs)


def has_yt_dlp_module():
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return False


_BPM_LO, _BPM_HI = 70, 175  # octave-fold range covers virtually all popular music

# Tonal hierarchy profiles (KK 1982, Temperley 2007, Shaath 2011) — normalised once at import
_MAJ_PROFILES = [p / p.sum() for p in [
    np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]),
    np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
    np.array([6.6, 2.0, 3.5, 2.3, 4.6, 3.9, 2.3, 5.4, 2.3, 3.7, 2.3, 3.4]),
]]
_MIN_PROFILES = [p / p.sum() for p in [
    np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]),
    np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    np.array([6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.1, 4.0, 2.7, 4.4, 3.2]),
]]

_AUDIO_MIMETYPES = {"wav": "audio/wav", "flac": "audio/flac"}


def _normalize_bpm(bpm: float) -> float:
    """Fold bpm into [_BPM_LO, _BPM_HI] by halving or doubling."""
    while bpm > 0 and bpm < _BPM_LO:
        bpm *= 2
    while bpm > _BPM_HI:
        bpm /= 2
    return bpm


def detect_tempo(y: np.ndarray, sr: int) -> int:
    """ACF-based BPM via librosa, folded into [70, 175]."""
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=onset, sr=sr, ac_size=8.0)
    )[0])
    return round(_normalize_bpm(tempo))


def _bass_tonic(y: np.ndarray, sr: int) -> int:
    """Pitch class with the most energy in the 60-300 Hz bass register."""
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= 60) & (freqs < 300)
    if not mask.any():
        return 0
    pc = np.round(librosa.hz_to_midi(freqs[mask])).astype(int) % 12
    energy = np.zeros(12)
    for i, p in enumerate(pc):
        energy[p] += float(S[mask][i].mean())
    return int(np.argmax(energy))


def detect_key(y: np.ndarray, sr: int):
    """
    Key detection: onset-weighted chroma_cens + three-profile KK/Temperley/Shaath
    ensemble. Bass-register tiebreaker for relative major/minor ambiguity.
    Returns (key_name, confidence, camelot_code).
    """
    hop = 512
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop, bins_per_octave=36)

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    n = min(chroma.shape[1], len(onset))
    w = np.sqrt(np.maximum(onset[:n], 0))
    w /= w.sum() + 1e-8
    chroma_agg = (chroma[:, :n] * w).sum(axis=1)
    chroma_agg /= chroma_agg.sum() + 1e-8

    major_scores = np.array([
        float(np.mean([np.corrcoef(chroma_agg, np.roll(p, i))[0, 1] for p in _MAJ_PROFILES]))
        for i in range(12)
    ])
    minor_scores = np.array([
        float(np.mean([np.corrcoef(chroma_agg, np.roll(p, i))[0, 1] for p in _MIN_PROFILES]))
        for i in range(12)
    ])

    bm = int(np.argmax(major_scores))
    bn = int(np.argmax(minor_scores))
    ms, ns = float(major_scores[bm]), float(minor_scores[bn])

    if (bm - bn) % 12 == 3 and abs(ms - ns) < 0.03:
        bt = _bass_tonic(y, sr)
        if bt == bn:
            ns += 0.01
        elif bt == bm:
            ms += 0.01

    all_sorted = sorted(list(major_scores) + list(minor_scores), reverse=True)
    confidence = float(np.clip((all_sorted[0] - all_sorted[1]) / (abs(all_sorted[0]) + 1e-8), 0.0, 1.0))

    if ms >= ns:
        return f"{NOTE_NAMES[bm]} Major", confidence, CAMELOT_MAJOR[bm]
    return f"{NOTE_NAMES[bn]} Minor", confidence, CAMELOT_MINOR[bn]


def sanitize_filename(s):
    return re.sub(r'[^\w\-]', '_', s)


def random_alnum_name(length=32):
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_output_basename(metadata, key_name="", bpm=0, source_name=""):
    # Song name: Shazam title → uploaded/source filename → short random
    title = sanitize_filename((metadata or {}).get("title") or "").strip("_")
    name  = title or sanitize_filename(source_name).strip("_") or random_alnum_name(8)
    # Key: "F# Major" → "Fs_Major"
    safe_key = key_name.replace("#", "s").replace(" ", "_") if key_name else ""
    bpm_part = f"{bpm}BPM" if bpm else ""
    return "_".join(p for p in [name, safe_key, bpm_part] if p)


def sanitize_tag_value(value):
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def build_unique_download_path(filename):
    candidate = DOWNLOADS_DIR / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(1, 10000):
        numbered = DOWNLOADS_DIR / f"{stem}_{idx}{suffix}"
        if not numbered.exists():
            return numbered
    return DOWNLOADS_DIR / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"


def parse_shazam_track(recognition):
    track = (recognition or {}).get("track") or {}
    if not track:
        return {}
    sections = track.get("sections") or []
    meta_section = next((s for s in sections if s.get("type") == "SONG"), {})
    metadata_pairs = meta_section.get("metadata") or []
    metadata_map = {
        (item.get("title") or "").strip().lower(): (item.get("text") or "").strip()
        for item in metadata_pairs
    }
    images = track.get("images") or {}
    return {
        "title": track.get("title") or metadata_map.get("title"),
        "artist": track.get("subtitle"),
        "album": metadata_map.get("album"),
        "genre": (track.get("genres") or {}).get("primary"),
        "shazam_url": track.get("url"),
        "cover_url": images.get("coverarthq") or images.get("coverart"),
    }


def identify_with_shazam(audio_path):
    project_dir = Path(__file__).resolve().parent
    shazam_python = os.getenv("SHAZAM_PYTHON", str(default_venv_python(".venv-shazam")))
    shazam_worker = os.getenv("SHAZAM_WORKER", str(project_dir / "shazam_worker.py"))

    if not Path(shazam_python).is_file():
        return {}, f"Shazam environment not found at {shazam_python}"
    if not Path(shazam_worker).is_file():
        return {}, f"Shazam worker script not found at {shazam_worker}"

    cmd = [shazam_python, shazam_worker, str(audio_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as exc:
        return {}, f"Shazam worker failed to start: {exc}"

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[:400]
        return {}, f"Shazam worker exited with code {proc.returncode}: {tail}"

    try:
        payload = json.loads(proc.stdout or "{}")
    except Exception as exc:
        return {}, f"Shazam worker returned invalid JSON: {exc}"

    if not payload.get("ok"):
        return {}, payload.get("error") or "Shazam did not return track metadata"

    metadata = payload.get("metadata") or {}
    if not metadata:
        return {}, "Shazam worker returned no metadata"
    return metadata, None


def download_album_art(cover_url, target_path):
    if not cover_url:
        return None
    try:
        urllib.request.urlretrieve(cover_url, str(target_path))
        if target_path.is_file() and target_path.stat().st_size > 0:
            return target_path
    except Exception:
        return None
    return None


def prepare_stem_input(job_id, source_wav, target_wav):
    """
    Build a lighter waveform for separation on constrained hosts.
    Trims and downsamples to reduce CPU/RAM pressure.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_wav),
        "-t",
        str(MAX_STEM_SECONDS),
        "-ar",
        "32000",
        "-ac",
        "2",
        str(target_wav),
    ]
    proc = run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to prepare stem input:\n{proc.stderr.strip()[-400:]}")
    if not target_wav.is_file() or target_wav.stat().st_size < 1024:
        raise RuntimeError("Prepared stem input file missing or too small.")
    log(job_id, f"Prepared stem input: {target_wav.name} ({target_wav.stat().st_size // 1024} KB)")


def run_stem_separation(job_id, wav_path, stems_mode, output_root):
    if stems_mode not in ("4", "6"):
        return None, [], {}, None

    model_candidates = ["htdemucs_6s"] if stems_mode == "6" else ["htdemucs_ft", "htdemucs", "mdx_extra_q"]
    last_error = None
    for model_name in model_candidates:
        try:
            return _run_stem_separation_with_model(job_id, wav_path, stems_mode, output_root, model_name)
        except Exception as exc:
            last_error = exc
            log(job_id, f"[separating] model {model_name} failed: {exc}")
            if model_name != model_candidates[-1]:
                log(job_id, f"[separating] retrying with fallback model {model_candidates[-1]}...")
    raise RuntimeError(str(last_error) if last_error else "Stem separation failed")


def _run_stem_separation_with_model(job_id, wav_path, stems_mode, output_root, model_name):
    device = resolve_demucs_device()
    update_msg = f"Separating into {stems_mode} stems with Demucs ({model_name}) on {device}..."
    jobs[job_id]["status"] = "separating"
    jobs[job_id]["message"] = update_msg
    log(job_id, f"[separating] {update_msg}")

    # HTDemucs Transformer models (htdemucs, htdemucs_6s) have a hard max segment
    # baked into their weights (~7.8s). Passing --segment above that is fatal.
    # MDX-Net models (mdx_extra_q, mdx_extra) have no such constraint and benefit
    # from larger segments that keep the GPU continuously fed.
    is_transformer = model_name.startswith("htdemucs")

    def build_cmd(chosen_device):
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n", model_name,
            "--device", chosen_device,
        ]
        if chosen_device == "cuda":
            cmd += ["--overlap", "0.25"]
            if not is_transformer:
                # MDX models: large segments saturate the GPU.
                cmd += ["--segment", "30"]
            # HTDemucs: omit --segment entirely so Demucs uses the model's own
            # trained default (7.8s for htdemucs_6s), avoiding the fatal cap error.
        elif chosen_device == "mps":
            cmd += ["--overlap", "0.25"]
            if not is_transformer:
                cmd += ["--segment", "20"]
        else:
            # CPU: keep segments small and dataloader workers low to avoid RAM pressure.
            cmd += ["-j", "1", "--overlap", "0.1"]
            if not is_transformer:
                cmd += ["--segment", "6"]
        cmd += ["-o", str(output_root), str(wav_path)]
        return cmd

    cmd = build_cmd(device)
    demucs_env = os.environ.copy()
    if device == "cuda":
        demucs_env.setdefault("TORCH_CUDNN_BENCHMARK", "1")
        alloc_conf, alloc_desc = build_cuda_alloc_conf()
        demucs_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
        log(job_id, f"[gpu] {alloc_desc}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=demucs_env,
    )
    start = time.time()
    last_log = start
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            now = time.time()
            if line:
                clean = line.strip()
                if clean:
                    output_lines.append(clean)
                    log(job_id, f"demucs: {clean[:240]}")
                    last_log = now
            elif proc.poll() is not None:
                break
            elif now - last_log >= 10:
                elapsed = int(now - start)
                jobs[job_id]["message"] = f"Separating stems... {elapsed}s elapsed"
                log(job_id, f"[separating] still running ({elapsed}s elapsed)")
                last_log = now
            else:
                time.sleep(0.2)
    finally:
        if proc.stdout:
            proc.stdout.close()
    return_code = proc.wait(timeout=10)
    if return_code != 0 and device != "cpu":
        tail = "\n".join(output_lines[-10:])
        log(job_id, f"[separating] {device} failed; retrying on cpu")
        log(job_id, f"[separating] {device} error tail: {tail[:500]}")
        cmd = build_cmd("cpu")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=demucs_env,
        )
        start = time.time()
        last_log = start
        output_lines = []
        try:
            while True:
                line = proc.stdout.readline() if proc.stdout else ""
                now = time.time()
                if line:
                    clean = line.strip()
                    if clean:
                        output_lines.append(clean)
                        log(job_id, f"demucs: {clean[:240]}")
                        last_log = now
                elif proc.poll() is not None:
                    break
                elif now - last_log >= 10:
                    elapsed = int(now - start)
                    jobs[job_id]["message"] = f"Separating stems on cpu... {elapsed}s elapsed"
                    log(job_id, f"[separating] still running on cpu ({elapsed}s elapsed)")
                    last_log = now
                else:
                    time.sleep(0.2)
        finally:
            if proc.stdout:
                proc.stdout.close()
        return_code = proc.wait(timeout=10)
    if return_code != 0:
        tail = "\n".join(output_lines[-10:])
        code_note = f"Demucs exit code: {return_code}."
        if return_code in (-9, 137):
            code_note += " Process likely killed by OOM; try 4 stems with lower memory load."
        raise RuntimeError(
            "Stem separation failed. Verify Demucs CLI args/model availability and retry.\n"
            f"{code_note}\n{tail}"
        )

    stem_dir = output_root / model_name / wav_path.stem
    if not stem_dir.is_dir():
        raise RuntimeError("Demucs completed but no stem directory was produced.")

    required_stems = ["drums", "bass", "vocals", "other"]
    if stems_mode == "6":
        required_stems.extend(["guitar", "piano"])

    produced = {}
    for p in stem_dir.iterdir():
        if not p.is_file():
            continue
        produced[p.stem.lower()] = p.name

    missing = [stem for stem in required_stems if stem not in produced]
    if missing:
        raise RuntimeError(
            f"Stem files missing after separation: {', '.join(missing)}. "
            f"Produced files: {', '.join(sorted(produced.values())) or 'none'}"
        )

    zip_base = output_root / f"stems_{stems_mode}"
    zip_file = shutil.make_archive(str(zip_base), "zip", root_dir=stem_dir)
    ordered_files = [produced[s] for s in required_stems if s in produced]
    stem_file_map = {s: produced[s] for s in required_stems if s in produced}
    return Path(zip_file), ordered_files, stem_file_map, stem_dir


def log(job_id, msg):
    print(f"[{job_id[:8]}] {msg}", flush=True)
    jobs[job_id].setdefault("log", []).append(msg)


def encode_output(job_id, wav_path, out_path, format_type, lufs, metadata_tags, album_art_path, update_fn):
    """Encode wav_path → out_path in the requested format, with optional LUFS normalization."""
    lufs_filter = ["-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11"] if lufs else []

    if format_type == "flac":
        update_fn("encoding", "Encoding FLAC...")
        cmd = ["ffmpeg", "-y", "-i", str(wav_path)] + lufs_filter + metadata_tags + ["-codec:a", "flac", str(out_path)]
        r = run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"FLAC encoding failed:\n{r.stderr.strip()[-400:]}")

    elif format_type == "wav":
        cmd = ["ffmpeg", "-y", "-i", str(wav_path)] + lufs_filter + metadata_tags + ["-c:a", "pcm_s16le", str(out_path)]
        r = run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"WAV finalize failed:\n{r.stderr.strip()[-400:]}")

    else:  # mp3
        update_fn("encoding", "Encoding final MP3...")
        cmd = ["ffmpeg", "-y", "-i", str(wav_path)]
        if album_art_path and album_art_path.is_file():
            cmd += ["-i", str(album_art_path),
                    "-map", "0:a", "-map", "1:v",
                    "-c:v", "mjpeg", "-id3v2_version", "3",
                    "-metadata:s:v", "title=Album cover",
                    "-metadata:s:v", "comment=Cover (front)"]
        cmd += lufs_filter + metadata_tags + ["-codec:a", "libmp3lame", "-qscale:a", "0", str(out_path)]
        r = run(cmd, capture_output=True, text=True, timeout=300)
        log(job_id, f"ffmpeg encode exit={r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(f"Encoding failed:\n{r.stderr.strip()[-400:]}")

    if not out_path.is_file() or out_path.stat().st_size < 1024:
        raise RuntimeError(f"{format_type.upper()} output missing or too small after encoding.")


def process_job(job_id, url, format_type="mp3", stems_mode="none", lufs=None, source_name=""):
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    def update(status, message):
        jobs[job_id]["status"] = status
        jobs[job_id]["message"] = message
        log(job_id, f"[{status}] {message}")

    try:
        # STEP 1: DOWNLOAD
        update("downloading", "Downloading media...")
        raw_template = str(job_dir / "raw.%(ext)s")
        dl = subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "-x",
                "--audio-format", "best",
                "--audio-quality", "0",
                "-o", raw_template,
                "--no-warnings",
                "--print", "after_move:%(title)s",
                "--print", "after_move:filepath",
                "--extractor-args", "youtube:innertube_api_host=yewtu.be",
                url,
            ],
            capture_output=True, text=True, timeout=300,
        )

        log(job_id, f"yt-dlp stdout: {dl.stdout.strip()}")
        if dl.stderr.strip():
            log(job_id, f"yt-dlp stderr: {dl.stderr.strip()[:500]}")
        if dl.returncode != 0:
            raise RuntimeError(
                f"Download failed (exit {dl.returncode}).\n"
                f"{dl.stderr.strip()[:400] or dl.stdout.strip()[:400]}"
            )

        lines = [l.strip() for l in dl.stdout.strip().splitlines() if l.strip()]
        printed_path = lines[-1] if lines else ""
        source_name  = lines[-2] if len(lines) >= 2 else ""
        if printed_path and Path(printed_path).is_file():
            raw_file = Path(printed_path)
        else:
            candidates = [p for p in job_dir.glob("raw.*") if p.suffix.lower() != ".part"]
            if not candidates:
                raise RuntimeError(
                    f"Download appeared to succeed but no file found in {job_dir}. "
                    f"Present: {list(job_dir.iterdir())}"
                )
            raw_file = candidates[0]
        log(job_id, f"Downloaded: {raw_file.name} ({raw_file.stat().st_size // 1024} KB)")

        # STEP 2: CONVERT TO WAV
        update("converting", "Converting to high-quality WAV...")
        wav_path = job_dir / "audio.wav"
        conv = run(
            [
                "ffmpeg", "-y", "-i", str(raw_file),
                "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", str(wav_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        log(job_id, f"ffmpeg convert exit={conv.returncode}")
        if conv.returncode != 0:
            raise RuntimeError(f"Conversion failed:\n{conv.stderr.strip()[-400:]}")
        if not wav_path.is_file() or wav_path.stat().st_size < 1024:
            raise RuntimeError("WAV file missing or too small after conversion.")
        log(job_id, f"WAV size: {wav_path.stat().st_size // 1024} KB")

        # STEP 3: LOOKUP METADATA
        update("metadata", "Extracting track metadata from Shazam...")
        shazam_meta, shazam_error = identify_with_shazam(wav_path)
        if shazam_error:
            log(job_id, f"Metadata note: {shazam_error}")
        else:
            log(job_id, "Shazam metadata found")

        album_art_path = None
        if shazam_meta.get("cover_url"):
            album_art_path = download_album_art(shazam_meta["cover_url"], job_dir / "cover.jpg")
            if album_art_path:
                log(job_id, f"Album art downloaded ({album_art_path.stat().st_size // 1024} KB)")
            else:
                log(job_id, "Album art download failed")

        # STEP 4: ANALYSE
        update("analyzing", "Analyzing key and tempo...")
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True, duration=60)
        log(job_id, f"Loaded {len(y)/sr:.1f}s at {sr} Hz")
        bpm = detect_tempo(y, sr)
        key_name, confidence, camelot_code = detect_key(y, sr)
        log(job_id, f"Key: {key_name} ({camelot_code})  BPM: {bpm}  Confidence: {confidence*100:.1f}%")
        output_base = build_output_basename(shazam_meta, key_name, bpm, source_name)

        metadata_tags = []
        for key, value in (
            ("title", shazam_meta.get("title")),
            ("artist", shazam_meta.get("artist")),
            ("album", shazam_meta.get("album")),
            ("genre", shazam_meta.get("genre")),
        ):
            cleaned = sanitize_tag_value(value)
            if cleaned:
                metadata_tags.extend(["-metadata", f"{key}={cleaned}"])

        # STEP 5: FINALIZE OUTPUT
        ext = "flac" if format_type == "flac" else ("wav" if format_type == "wav" else "mp3")
        out_filename = f"{output_base}.{ext}"
        out_path = job_dir / out_filename
        encode_output(job_id, wav_path, out_path, format_type, lufs, metadata_tags, album_art_path, update)

        log(job_id, f"Output: {out_path.stat().st_size // 1024} KB -> {out_filename}")

        stems_zip = None
        stems_files = []
        stem_file_map = {}
        stems_dir = None
        if stems_mode in ("4", "6"):
            demucs_out_dir = job_dir / "demucs_out"
            stem_input = wav_path
            if LOW_RESOURCE_MODE:
                update("separating", f"Preparing low-resource stem input (first {MAX_STEM_SECONDS}s @ 32kHz)...")
                reduced_wav = job_dir / "audio_stems.wav"
                prepare_stem_input(job_id, wav_path, reduced_wav)
                stem_input = reduced_wav
            stems_zip, stems_files, stem_file_map, stems_dir = run_stem_separation(job_id, stem_input, stems_mode, demucs_out_dir)
            log(job_id, f"Stem ZIP ready: {stems_zip.name}")

        raw_file.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        if album_art_path:
            album_art_path.unlink(missing_ok=True)

        jobs[job_id].update(
            {
                "status": "done",
                "message": "Complete!",
                "filename": out_filename,
                "filepath": str(out_path),
                "key": key_name,
                "camelot": camelot_code,
                "tempo": bpm,
                "duration_secs": round(len(y) / sr, 2),
                "confidence": round(confidence * 100, 1),
                "format": format_type,
                "lufs": lufs,
                "stems_mode": stems_mode,
                "stems_zip_filename": stems_zip.name if stems_zip else None,
                "stems_zip_path": str(stems_zip) if stems_zip else None,
                "stems_files": stems_files,
                "stems_dir": str(stems_dir) if stems_dir else None,
                "stems_map": stem_file_map,
                "metadata": {
                    "title": shazam_meta.get("title"),
                    "artist": shazam_meta.get("artist"),
                    "album": shazam_meta.get("album"),
                    "genre": shazam_meta.get("genre"),
                    "shazam_url": shazam_meta.get("shazam_url"),
                    "album_art_url": shazam_meta.get("cover_url"),
                    "album_art_embedded": bool(format_type == "mp3" and album_art_path),
                },
            }
        )

    except Exception as e:
        msg = str(e)
        log(job_id, f"ERROR: {msg}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = msg


def process_file_job(job_id, raw_file, format_type="mp3", stems_mode="none", lufs=None, source_name=""):
    job_dir = raw_file.parent

    def update(status, message):
        jobs[job_id]["status"] = status
        jobs[job_id]["message"] = message
        log(job_id, f"[{status}] {message}")

    try:
        log(job_id, f"Processing uploaded file: {raw_file.name} ({raw_file.stat().st_size // 1024} KB)")

        update("converting", "Converting to high-quality WAV...")
        wav_path = job_dir / "audio.wav"
        conv = run(
            ["ffmpeg", "-y", "-i", str(raw_file), "-ar", "44100", "-ac", "2", "-sample_fmt", "s16", str(wav_path)],
            capture_output=True, text=True, timeout=300,
        )
        log(job_id, f"ffmpeg convert exit={conv.returncode}")
        if conv.returncode != 0:
            raise RuntimeError(f"Conversion failed:\n{conv.stderr.strip()[-400:]}")
        if not wav_path.is_file() or wav_path.stat().st_size < 1024:
            raise RuntimeError("WAV file missing or too small after conversion.")
        log(job_id, f"WAV size: {wav_path.stat().st_size // 1024} KB")

        update("metadata", "Extracting track metadata from Shazam...")
        shazam_meta, shazam_error = identify_with_shazam(wav_path)
        if shazam_error:
            log(job_id, f"Metadata note: {shazam_error}")
        else:
            log(job_id, "Shazam metadata found")

        album_art_path = None
        if shazam_meta.get("cover_url"):
            album_art_path = download_album_art(shazam_meta["cover_url"], job_dir / "cover.jpg")
            if album_art_path:
                log(job_id, f"Album art downloaded ({album_art_path.stat().st_size // 1024} KB)")
            else:
                log(job_id, "Album art download failed")

        update("analyzing", "Analyzing key and tempo...")
        y, sr = librosa.load(str(wav_path), sr=22050, mono=True, duration=60)
        log(job_id, f"Loaded {len(y)/sr:.1f}s at {sr} Hz")
        bpm = detect_tempo(y, sr)
        key_name, confidence, camelot_code = detect_key(y, sr)
        log(job_id, f"Key: {key_name} ({camelot_code})  BPM: {bpm}  Confidence: {confidence*100:.1f}%")
        output_base = build_output_basename(shazam_meta, key_name, bpm, source_name)

        metadata_tags = []
        for key, value in (
            ("title", shazam_meta.get("title")),
            ("artist", shazam_meta.get("artist")),
            ("album", shazam_meta.get("album")),
            ("genre", shazam_meta.get("genre")),
        ):
            cleaned = sanitize_tag_value(value)
            if cleaned:
                metadata_tags.extend(["-metadata", f"{key}={cleaned}"])

        ext = "flac" if format_type == "flac" else ("wav" if format_type == "wav" else "mp3")
        out_filename = f"{output_base}.{ext}"
        out_path = job_dir / out_filename
        encode_output(job_id, wav_path, out_path, format_type, lufs, metadata_tags, album_art_path, update)

        log(job_id, f"Output: {out_path.stat().st_size // 1024} KB -> {out_filename}")

        stems_zip = None
        stems_files = []
        stem_file_map = {}
        stems_dir = None
        if stems_mode in ("4", "6"):
            demucs_out_dir = job_dir / "demucs_out"
            stem_input = wav_path
            if LOW_RESOURCE_MODE:
                update("separating", f"Preparing low-resource stem input (first {MAX_STEM_SECONDS}s @ 32kHz)...")
                reduced_wav = job_dir / "audio_stems.wav"
                prepare_stem_input(job_id, wav_path, reduced_wav)
                stem_input = reduced_wav
            stems_zip, stems_files, stem_file_map, stems_dir = run_stem_separation(job_id, stem_input, stems_mode, demucs_out_dir)
            log(job_id, f"Stem ZIP ready: {stems_zip.name}")

        raw_file.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)
        if album_art_path:
            album_art_path.unlink(missing_ok=True)

        jobs[job_id].update({
            "status": "done",
            "message": "Complete!",
            "filename": out_filename,
            "filepath": str(out_path),
            "key": key_name,
            "camelot": camelot_code,
            "tempo": bpm,
            "duration_secs": round(len(y) / sr, 2),
            "confidence": round(confidence * 100, 1),
            "format": format_type,
            "lufs": lufs,
            "stems_mode": stems_mode,
            "stems_zip_filename": stems_zip.name if stems_zip else None,
            "stems_zip_path": str(stems_zip) if stems_zip else None,
            "stems_files": stems_files,
            "stems_dir": str(stems_dir) if stems_dir else None,
            "stems_map": stem_file_map,
            "metadata": {
                "title": shazam_meta.get("title"),
                "artist": shazam_meta.get("artist"),
                "album": shazam_meta.get("album"),
                "genre": shazam_meta.get("genre"),
                "shazam_url": shazam_meta.get("shazam_url"),
                "album_art_url": shazam_meta.get("cover_url"),
                "album_art_embedded": bool(format_type == "mp3" and album_art_path),
            },
        })

    except Exception as e:
        msg = str(e)
        log(job_id, f"ERROR: {msg}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = msg


# ROUTES

@app.route("/")
def index():
    if is_free_tier():
        return send_from_directory("static", "web.html")
    return send_from_directory("static", "index.html")


@app.route("/founders")
def founders():
    return send_from_directory("static", "founders.html")


@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    format_type = data.get("format", "mp3").lower()
    if format_type not in ("mp3", "wav", "flac"):
        format_type = "mp3"
    stems_mode = str(data.get("stems", "none")).lower()
    if stems_mode not in ("none", "4", "6"):
        stems_mode = "none"
    raw_lufs = data.get("lufs")
    lufs = int(raw_lufs) if raw_lufs and str(raw_lufs).lstrip("-").isdigit() else None
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    if is_free_tier():
        format_type = "mp3"
        stems_mode  = "none"
        lufs        = None

    if not has_yt_dlp_module():
        return jsonify({
            "error": "'yt_dlp' python module not found in current environment. "
                     "Install dependencies in .venv-main (./setup_envs.sh update)."
        }), 500
    if not find_tool("ffmpeg"):
        return jsonify({
            "error": "'ffmpeg' is not installed or not on your PATH. "
                     "Install ffmpeg or place ffmpeg binary next to server.py."
        }), 500

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "queued",
        "message": "Queued...",
        "filename": None,
        "log": [],
        "format": format_type,
        "stems_mode": stems_mode,
    }
    print(f"[{job_id[:8]}] Queued: {url} (format: {format_type}, stems: {stems_mode}, lufs: {lufs})", flush=True)

    threading.Thread(target=process_job, args=(job_id, url, format_type, stems_mode, lufs), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/submit-file", methods=["POST"])
def submit_file():
    uploaded = request.files.get("file")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file provided"}), 400

    format_type = (request.form.get("format") or "mp3").lower()
    if format_type not in ("mp3", "wav", "flac"):
        format_type = "mp3"
    stems_mode = str(request.form.get("stems") or "none").lower()
    if stems_mode not in ("none", "4", "6"):
        stems_mode = "none"
    raw_lufs = request.form.get("lufs")
    lufs = int(raw_lufs) if raw_lufs and str(raw_lufs).lstrip("-").isdigit() else None

    if not find_tool("ffmpeg"):
        return jsonify({"error": "'ffmpeg' is not installed or not on your PATH."}), 500

    if is_free_tier():
        format_type = "mp3"
        stems_mode  = "none"
        lufs        = None

    job_id = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded.filename).suffix or ".audio"
    raw_file = job_dir / f"raw{suffix}"
    uploaded.save(str(raw_file))

    jobs[job_id] = {
        "status": "queued",
        "message": "Queued...",
        "filename": None,
        "log": [],
        "format": format_type,
        "stems_mode": stems_mode,
    }
    source_name = Path(uploaded.filename).stem if uploaded.filename else ""
    print(f"[{job_id[:8]}] Queued upload: {uploaded.filename} (format: {format_type}, stems: {stems_mode}, lufs: {lufs})", flush=True)

    threading.Thread(target=process_file_job, args=(job_id, raw_file, format_type, stems_mode, lufs, source_name), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


def _copy_to_downloads(src: str, name: str) -> None:
    try:
        shutil.copy2(src, build_unique_download_path(name))
    except Exception:
        pass


@app.route("/api/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"Job not ready (status: {job['status']})"}), 404
    filepath = job.get("filepath")
    if not filepath or not Path(filepath).is_file():
        return jsonify({"error": "Output file is missing from disk"}), 500
    return send_file(
        filepath,
        as_attachment=True,
        download_name=job["filename"],
        mimetype=_AUDIO_MIMETYPES.get(job.get("format"), "audio/mpeg"),
    )


@app.route("/api/stream/<job_id>")
def stream(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"Job not ready (status: {job['status']})"}), 404
    filepath = job.get("filepath")
    if not filepath or not Path(filepath).is_file():
        return jsonify({"error": "Output file is missing from disk"}), 500
    return send_file(
        filepath,
        as_attachment=False,
        download_name=job["filename"],
        mimetype=_AUDIO_MIMETYPES.get(job.get("format"), "audio/mpeg"),
    )


@app.route("/api/download-stems/<job_id>")
def download_stems(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    stems_path = job.get("stems_zip_path")
    stems_name = job.get("stems_zip_filename")
    if not stems_path or not stems_name:
        return jsonify({"error": "No stem package available for this job"}), 404
    if not Path(stems_path).is_file():
        return jsonify({"error": "Stem package is missing from disk"}), 500
    return send_file(stems_path, as_attachment=True, download_name=stems_name, mimetype="application/zip")


@app.route("/api/stream-stem/<job_id>/<stem_name>")
def stream_stem(job_id, stem_name):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    if job["status"] != "done":
        return jsonify({"error": f"Job not ready (status: {job['status']})"}), 404
    stem_map = job.get("stems_map") or {}
    safe_stem = (stem_name or "").strip().lower()
    filename = stem_map.get(safe_stem)
    stems_dir = job.get("stems_dir")
    if not filename or not stems_dir:
        return jsonify({"error": f"Stem '{safe_stem}' not available for this job"}), 404
    stem_path = Path(stems_dir) / filename
    if not stem_path.is_file():
        return jsonify({"error": "Stem file is missing from disk"}), 500
    return send_file(
        str(stem_path),
        as_attachment=False,
        download_name=filename,
        mimetype="audio/wav",
    )


# FL STUDIO INTEGRATION

@app.route("/api/fl/find")
def fl_find():
    if is_free_tier():
        return jsonify({"error": "Not available in free tier"}), 403
    path = find_fl_studio()
    return jsonify({"path": path, "found": path is not None})


RECORDING_TEMPLATE = Path(__file__).parent / "assets" / "recording_template.flp"
FL_SESSIONS_DIR = Path(tempfile.gettempdir()) / "url2audio_sessions"
FL_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _fl_playlist_item_bytes(position=0, item_index=0, length=98304,
                             track_rvidx=499, group=0, item_flags=64):
    """Build one 60-byte new-format PlaylistEvent item (FL Studio 21+)."""
    import struct as _struct
    data = bytearray(60)
    _struct.pack_into('<I', data, 0,  position)
    _struct.pack_into('<H', data, 4,  0x5000)        # pattern_base always 20480
    _struct.pack_into('<H', data, 6,  item_index)
    _struct.pack_into('<I', data, 8,  length)
    _struct.pack_into('<H', data, 12, track_rvidx)
    _struct.pack_into('<H', data, 14, group)
    data[16] = 120; data[17] = 0                     # _u1 always (120, 0)
    _struct.pack_into('<H', data, 18, item_flags)
    data[20:24] = bytes([64, 100, 128, 128])          # _u2 always these
    # start_offset + end_offset = 0.0 each (already zero)
    # _u3 = 28 bytes zeros (already zero)
    return bytes(data)


def _build_fl_session(beat_path: Path, bpm: int = 0, duration_secs: float = 0) -> Path:
    """Clone recording_template.flp, inject beat path, BPM, and playlist clip."""
    import warnings

    try:
        import pyflp
        from pyflp._events import UnicodeEvent, U32Event, IndexedEvent
        from pyflp.channel import ChannelID
        from pyflp.project import ProjectID
        from pyflp.arrangement import ArrangementID, PlaylistEvent
    except ImportError:
        raise RuntimeError("pyflp not installed — run: pip install pyflp")

    if not RECORDING_TEMPLATE.is_file():
        raise FileNotFoundError(f"Recording template not found: {RECORDING_TEMPLATE}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = pyflp.parse(str(RECORDING_TEMPLATE))

    # ── inject beat path into Sampler channel ──────────────────────────────
    sampler = next((ch for ch in project.channels if type(ch).__name__ == "Sampler"), None)
    if sampler is None:
        raise RuntimeError("No Sampler channel found in recording template")

    channel_iid = int(sampler.iid) if sampler.iid is not None else 0
    beat_str = str(beat_path)

    if sampler.sample_path is not None:
        sampler.sample_path = beat_path
    else:
        beat_bytes = beat_str.encode("utf-16-le") + b"\x00\x00"
        ev = UnicodeEvent(ChannelID.SamplePath, beat_bytes)
        max_r = max((ie.r for ie in sampler.events.lst), default=0)
        new_ie = IndexedEvent(r=max_r + 1, e=ev)
        sampler.events.lst.add(new_ie)
        if sampler.events.root is not sampler.events:
            sampler.events.root.lst.add(new_ie)

    # ── inject BPM ─────────────────────────────────────────────────────────
    if bpm and 10 <= bpm <= 522:
        if project.tempo is not None:
            try:
                project.tempo = int(bpm)
            except Exception:
                pass
        else:
            tempo_val = int(bpm) * 1000
            ev = U32Event(ProjectID.Tempo, tempo_val.to_bytes(4, "little"))
            max_r = max((ie.r for ie in project.events.lst), default=0)
            project.events.lst.add(IndexedEvent(r=max_r + 1, e=ev))

    # ── inject audio clip onto Playlist track 1 ────────────────────────────
    # Calculate length in PPQ ticks.  PPQ=96 is the FL Studio default.
    ppq = int(project.ppq) if project.ppq else 96
    if duration_secs and bpm:
        beat_length_ppq = int(duration_secs * bpm / 60 * ppq)
    else:
        beat_length_ppq = ppq * 4 * 256  # 256 bars fallback

    item_bytes = _fl_playlist_item_bytes(
        position=0,
        item_index=channel_iid,
        length=beat_length_ppq,
        track_rvidx=499,          # Track 1 (first track), reversed index for 500 max tracks
    )
    pl_ev = PlaylistEvent(ArrangementID.Playlist, item_bytes)

    # Insert right after ArrangementID.New (WORD+35 = 99)
    arr_new_r = next(
        (ie.r for ie in project.events.lst if int(ie.e.id) == int(ArrangementID.New)),
        max((ie.r for ie in project.events.lst), default=0),
    )
    project.events.lst.add(IndexedEvent(r=arr_new_r + 0.5, e=pl_ev))

    session_path = FL_SESSIONS_DIR / f"session_{uuid.uuid4().hex[:8]}.flp"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyflp.save(project, str(session_path))

    return session_path





@app.route("/api/fl/open", methods=["POST"])
def fl_open():
    if is_free_tier():
        return jsonify({"error": "Not available in free tier"}), 403
    data      = request.get_json(force=True)
    job_id    = data.get("job_id", "")
    stem_name = data.get("stem")

    fl_path = find_fl_studio()
    if not fl_path:
        return jsonify({"error": "FL Studio not found. Install FL Studio and try again."}), 404

    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if stem_name:
        stems_dir = job.get("stems_dir")
        stems_map = job.get("stems_map") or {}
        filename  = stems_map.get(stem_name.lower().strip())
        if not stems_dir or not filename:
            return jsonify({"error": f"Stem '{stem_name}' not available"}), 404
        target = Path(stems_dir) / filename
    else:
        target = Path(job.get("filepath", ""))

    if not target.is_file():
        return jsonify({"error": "Audio file not found on disk"}), 404

    bpm = int(job.get("tempo") or 0)
    duration_secs = float(job.get("duration_secs") or 0)
    try:
        session_flp = _build_fl_session(target, bpm=bpm, duration_secs=duration_secs)
    except Exception as exc:
        return jsonify({"error": f"Could not build FL session: {exc}"}), 500

    launch_fl_studio(fl_path, session_flp)
    return jsonify({"ok": True, "opened": str(target), "session": str(session_flp), "bpm": bpm})


# DEV UTILITIES

@app.route("/api/dev/tier-status")
def dev_tier_status():
    return jsonify({"free_tier": is_free_tier(), "overridden": _dev_tier_override is not None})

@app.route("/api/dev/toggle-tier", methods=["POST"])
def dev_toggle_tier():
    global _dev_tier_override
    _dev_tier_override = not is_free_tier()
    return jsonify({"free_tier": _dev_tier_override})


# VOCAL CHAIN STORE

@app.route("/api/chain-path/<chain_id>")
def chain_path(chain_id):
    if is_free_tier():
        return jsonify({"error": "Not available in free tier"}), 403
    chain = next((c for c in VOCAL_CHAINS if c["id"] == chain_id), None)
    if not chain:
        return jsonify({"error": "Unknown chain"}), 404
    fst = CHAINS_DIR / chain["file"]
    if fst.is_file():
        return jsonify({"path": str(fst)})
    return jsonify({"error": "Chain not purchased yet"}), 404


@app.route("/api/chains")
def get_chains():
    owned = {p.stem for p in CHAINS_DIR.glob("*.fst")}
    result = []
    for c in VOCAL_CHAINS:
        entry = dict(c)
        entry["owned"] = c["id"] in owned
        result.append(entry)
    return jsonify(result)


@app.route("/api/buy-chain", methods=["POST"])
def buy_chain():
    data     = request.get_json(force=True)
    chain_id = data.get("chain_id", "")
    chain    = next((c for c in VOCAL_CHAINS if c["id"] == chain_id), None)
    if not chain:
        return jsonify({"error": "Chain not found"}), 404

    # Free chains: copy immediately, no Stripe needed
    if chain.get("free"):
        fst_src = CHAINS_ASSET / chain["file"]
        fst_dst = CHAINS_DIR  / chain["file"]
        if fst_src.is_file():
            shutil.copy2(str(fst_src), str(fst_dst))
        return jsonify({"ok": True, "free": True})

    if not STRIPE_SECRET or not CHAIN_PRICE_ID:
        return jsonify({"error": "Store not configured — add STRIPE_SECRET_KEY and CHAIN_PRICE_ID"}), 503
    try:
        import stripe as _stripe  # type: ignore
        _stripe.api_key = STRIPE_SECRET
        session = _stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": CHAIN_PRICE_ID, "quantity": 1}],
            mode="payment",
            success_url=(request.host_url
                         + f"chain-success?session_id={{CHECKOUT_SESSION_ID}}&chain_id={chain_id}"),
            cancel_url=request.host_url,
            metadata={"chain_id": chain_id},
        )
        return jsonify({"url": session.url})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/buy-cart", methods=["POST"])
def buy_cart():
    data      = request.get_json(force=True)
    chain_ids = data.get("chain_ids", [])
    chains    = [c for c in VOCAL_CHAINS if c["id"] in chain_ids]
    if not chains:
        return jsonify({"error": "No valid chains in cart"}), 400

    free_chains = [c for c in chains if c.get("free")]
    paid_chains = [c for c in chains if not c.get("free")]

    for c in free_chains:
        fst_src = CHAINS_ASSET / c["file"]
        fst_dst = CHAINS_DIR   / c["file"]
        if fst_src.is_file() and not fst_dst.is_file():
            shutil.copy2(str(fst_src), str(fst_dst))

    if not paid_chains:
        return jsonify({"ok": True, "free": True})

    if not STRIPE_SECRET or not CHAIN_PRICE_ID:
        return jsonify({"error": "Store not configured — add STRIPE_SECRET_KEY and CHAIN_PRICE_ID"}), 503

    try:
        import stripe as _stripe  # type: ignore
        _stripe.api_key = STRIPE_SECRET
        cart_param = ",".join(c["id"] for c in paid_chains)
        session = _stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": CHAIN_PRICE_ID, "quantity": len(paid_chains)}],
            mode="payment",
            success_url=(request.host_url
                         + f"chain-success?session_id={{CHECKOUT_SESSION_ID}}&cart={cart_param}"),
            cancel_url=request.host_url,
            metadata={"cart_ids": cart_param},
        )
        return jsonify({"url": session.url})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/chain-success")
def chain_success():
    session_id   = request.args.get("session_id", "")
    chain_id     = request.args.get("chain_id", "")      # legacy single-chain flow
    cart_param   = request.args.get("cart", "")          # new multi-chain cart flow
    if not session_id or not STRIPE_SECRET:
        return "Invalid request", 400

    # Resolve which chain IDs to deliver
    if cart_param:
        deliver_ids = [cid.strip() for cid in cart_param.split(",") if cid.strip()]
    elif chain_id:
        deliver_ids = [chain_id]
    else:
        return "No chains specified", 400

    chains_to_deliver = [c for c in VOCAL_CHAINS if c["id"] in deliver_ids]
    if not chains_to_deliver:
        return "Unknown chain(s)", 400

    try:
        import stripe as _stripe  # type: ignore
        _stripe.api_key = STRIPE_SECRET
        session = _stripe.checkout.Session.retrieve(session_id)
        if session.payment_status != "paid":
            return "Payment not confirmed", 402

        names = []
        for chain in chains_to_deliver:
            fst_src = CHAINS_ASSET / chain["file"]
            fst_dst = CHAINS_DIR   / chain["file"]
            if fst_src.is_file():
                shutil.copy2(str(fst_src), str(fst_dst))
            names.append(chain["name"])

        names_html = ", ".join(f"<strong>{n}</strong>" for n in names)
        plural     = "Chains" if len(names) > 1 else "Chain"
        return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Chains Unlocked</title>
<style>body{{background:#0a0a0f;color:#e8e8f0;font-family:'Segoe UI',sans-serif;
display:flex;align-items:center;justify-content:center;min-height:100vh;margin:0}}
.box{{text-align:center;padding:48px;background:#111118;border:1px solid #1e1e2e;border-radius:16px;max-width:480px}}
h1{{color:#753BBD;margin-bottom:12px}}p{{color:#aaa;margin-bottom:32px;line-height:1.6}}
.tag{{background:rgba(117,59,189,0.15);border:1px solid rgba(117,59,189,0.3);padding:4px 12px;
border-radius:20px;font-size:0.82rem;color:#9d4edd}}</style></head>
<body><div class="box">
<h1>&#127925; Vocal {plural} Unlocked</h1>
<p>{names_html}<br>{"have" if len(names) > 1 else "has"} been added to your library.<br>
Open URL2AUDIO, go to the <strong>Store → My Chains</strong> tab,<br>
and drag a preset directly into your FL Studio mixer channel.</p>
<span class="tag">Saved to My Chains</span>
</div></body></html>"""
    except Exception as exc:
        return str(exc), 500


# STRIPE / APP DOWNLOAD ROUTES

@app.route("/buy", methods=["POST"])
def buy():
    if not STRIPE_SECRET or not STRIPE_PRICE_ID:
        return jsonify({"error": "Store not configured on server"}), 503
    try:
        import stripe as _stripe  # type: ignore
        _stripe.api_key = STRIPE_SECRET
        session = _stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            mode="payment",
            success_url=request.host_url + "success?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.host_url + "#download",
        )
        return jsonify({"url": session.url})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/success")
def success():
    session_id = request.args.get("session_id", "")
    if not session_id or not STRIPE_SECRET:
        return "Invalid request", 400
    try:
        import stripe as _stripe  # type: ignore
        _stripe.api_key = STRIPE_SECRET
        session = _stripe.checkout.Session.retrieve(session_id)
        if session.payment_status != "paid":
            return "Payment not confirmed", 402
        token = secrets.token_urlsafe(32)
        with _token_lock:
            _download_tokens[token] = {
                "expires": time.time() + 86400,
                "path": APP_DOWNLOAD_PATH,
                "uses_left": 3,
            }
        download_url = f"/download/{token}"
        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>URL2AUDIO — Download Ready</title>
<style>
  body{{background:#0a0a0f;color:#e8e8f0;font-family:'Segoe UI',sans-serif;display:flex;
       align-items:center;justify-content:center;min-height:100vh;margin:0}}
  .box{{text-align:center;padding:48px;background:#111118;border:1px solid #1e1e2e;border-radius:16px;max-width:480px}}
  h1{{color:#753BBD;margin-bottom:12px}}
  p{{color:#aaa;margin-bottom:32px;line-height:1.6}}
  a{{display:inline-block;background:linear-gradient(135deg,#753BBD,#9d4edd);color:#fff;
     text-decoration:none;padding:14px 32px;border-radius:8px;font-weight:700;font-size:1rem}}
  small{{display:block;margin-top:20px;color:#555;font-size:0.8rem}}
</style></head>
<body><div class="box">
  <h1>Payment confirmed</h1>
  <p>Your download link is ready. It expires in 24 hours and allows 3 downloads.</p>
  <a href="{download_url}" download>Download URL2AUDIO</a>
  <small>Link: {request.host_url}download/{token}</small>
</div></body></html>"""
    except Exception as exc:
        return str(exc), 500


@app.route("/api/stripe/webhook", methods=["POST"])
def stripe_webhook():
    payload = request.get_data()
    sig = request.headers.get("Stripe-Signature", "")
    if not STRIPE_WEBHOOK_SECRET:
        return "", 200
    try:
        import stripe as _stripe  # type: ignore
        event = _stripe.Webhook.construct_event(payload, sig, STRIPE_WEBHOOK_SECRET)
    except Exception as exc:
        return str(exc), 400
    # Token is issued in /success; webhook is for logging/future extensibility
    if event.get("type") == "checkout.session.completed":
        print(f"[stripe] checkout.session.completed: {event['data']['object'].get('id')}", flush=True)
    return "", 200


@app.route("/download/<token>")
def app_download(token):
    with _token_lock:
        entry = _download_tokens.get(token)
        if not entry:
            return "Invalid or expired download link", 404
        if time.time() > entry["expires"]:
            _download_tokens.pop(token, None)
            return "Download link has expired", 410
        if entry["uses_left"] <= 0:
            return "Download limit reached for this link", 410
        entry["uses_left"] -= 1
        dl_path = entry["path"]
    if not dl_path or not Path(dl_path).is_file():
        return "Installer not found on server — contact support", 503
    return send_file(dl_path, as_attachment=True)


# CLEANUP

def cleanup():
    while True:
        time.sleep(3600)
        try:
            cutoff = time.time() - 7200
            for jid in list(jobs.keys()):
                try:
                    fp = jobs[jid].get("filepath")
                    if fp and Path(fp).is_file() and os.path.getmtime(fp) < cutoff:
                        shutil.rmtree(Path(fp).parent, ignore_errors=True)
                        jobs.pop(jid, None)
                except Exception:
                    pass
        except Exception:
            pass

threading.Thread(target=cleanup, daemon=True).start()

WAITLIST_FILE = Path(__file__).parent / "waitlist.csv"


@app.route("/api/waitlist", methods=["POST"])
def waitlist():
    data  = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    if not email or "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Invalid email"}), 400
    from datetime import datetime as _dt
    import csv as _csv
    signed_up_at = _dt.utcnow().isoformat()
    existed = WAITLIST_FILE.exists()
    with WAITLIST_FILE.open("a", newline="", encoding="utf-8") as f:
        if not existed:
            f.write("email,signed_up_at\n")
        w = _csv.writer(f)
        w.writerow([email, signed_up_at])
    print(f"[WAITLIST] {email} {signed_up_at}", flush=True)
    return jsonify({"ok": True})


if __name__ == "__main__":
    host = os.getenv("URL2AUDIO_HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("URL2AUDIO_PORT", "6969"))
    print(f"Work dir : {WORK_DIR}", flush=True)
    print(f"yt_dlp py: {'OK' if has_yt_dlp_module() else 'NOT FOUND'}", flush=True)
    print(f"ffmpeg   : {find_tool('ffmpeg') or 'NOT FOUND'}", flush=True)
    print(f"Demucs   : device={resolve_demucs_device()}  python={sys.executable}", flush=True)
    print(f"Listening: http://{host}:{port}", flush=True)
    app.run(host=host, port=port, debug=False)
