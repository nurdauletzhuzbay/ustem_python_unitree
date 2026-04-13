#!/usr/bin/env python3
"""
Temirbek Voice Assistant
Built-in microphone + Built-in speaker (via DDS AudioClient) + Azure Neural TTS + AlemAI STT/LLM

Pipeline: record → STT → LLM (streaming) → TTS per sentence → play
TTS synthesis of sentence N overlaps with LLM generation of sentence N+1.
"""

import os
import sys
import signal
import subprocess
import threading
import time
import select
import struct
import wave
import tempfile
import queue
import json
import html
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import requests

# Ensure repo root is on the path so unitree_sdk2py is importable
# regardless of how/where the script is invoked
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# wav.py lives in the same directory — needed for play_pcm_stream
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from wav import play_pcm_stream

# Load .env from the repo root
def _load_env():
    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ============================================================================
# CONFIGURATION
# ============================================================================
ALSA_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"    # JMTek USB microphone

AZURE_TTS_VOICE  = "kk-KZ-DauletNeural"           # Male Kazakh neural voice
AZURE_TTS_REGION = "eastus"

SAMPLE_RATE      = 16000
MAX_HISTORY_TURNS = 8
CHANNELS         = 1

# Flush LLM buffer to TTS when we hit a sentence delimiter and have enough text
SENTENCE_DELIMITERS = frozenset('.!?…:;')
MIN_CHUNK_LEN = 25  # chars

# ============================================================================
# GLOBAL STATE
# ============================================================================
g_running       = True
g_is_recording  = False
g_stop_recording = False

@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str

conversation_history: List[ConversationTurn] = []

# ============================================================================
# SIGNAL HANDLER
# ============================================================================
def signal_handler(sig, frame):
    global g_running, g_is_recording
    g_running = False
    g_is_recording = False
    print("\nShutting down...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# G1 AUDIO CLIENT — LED + built-in speaker via DDS
# ============================================================================
class G1AudioClient:
    def __init__(self, network_interface: str):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            ChannelFactoryInitialize(0, network_interface)
            self.client = AudioClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
            self.available = True
            print("G1 AudioClient initialized.")
        except Exception as e:
            print(f"Warning: G1 AudioClient unavailable ({e}). LED/speaker disabled.")
            self.available = False

    def led(self, r: int, g: int, b: int):
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception as e:
                print(f"LED error: {e}")

    def play_pcm(self, pcm_bytes: bytes, app_name: str = "temirbek"):
        """Stream raw PCM (16kHz mono 16-bit) to the built-in speaker via DDS."""
        if not self.available or not pcm_bytes or not g_running:
            return
        play_pcm_stream(self.client, list(pcm_bytes), app_name)

# ============================================================================
# SPEECH-TO-TEXT — AlemAI Whisper
# ============================================================================
def transcribe_audio(audio_pcm: List[int]) -> str:
    if not g_running:
        return ""

    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return ""

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        wav_path = tmp.name

    try:
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(struct.pack(f'{len(audio_pcm)}h', *audio_pcm))

        with open(wav_path, 'rb') as f:
            resp = requests.post(
                "https://llm.alem.ai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files={'file': ('audio.wav', f, 'audio/wav')},
                data={'model': 'speech-to-text-kk'},
                timeout=15
            )

        if resp.status_code == 200:
            return resp.json().get('text', '').strip()
        print(f"STT error: {resp.status_code}")
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

# ============================================================================
# LLM STREAMING — AlemAI (OpenAI-compatible SSE)
# ============================================================================
SYSTEM_PROMPT = (
    "Сен Темірбек деген Unitree G1 роботысың. Қазақша сөйлейсің.\n"
    "Мақсатың — адаммен шынайы, жылы әңгіме жүргізу. "
    "Жауап берген соң, өзің де қызықты сұрақ қой немесе тақырыпты дамыт — "
    "бос тұрма, диалогты ұстап тұр. "
    "Жауаптарың қысқа (1–3 сөйлем), табиғи, дос сияқты. Ресми тіл қолданба."
)

def stream_llm(user_text: str, sentence_queue: queue.Queue) -> str:
    """
    Streams the LLM response token by token.
    Pushes complete sentences into sentence_queue as they form.
    Puts None as a sentinel when done.
    Returns the full response text.
    """
    if not g_running:
        sentence_queue.put(None)
        return ""

    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        sentence_queue.put(None)
        return ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    start = max(0, len(conversation_history) - MAX_HISTORY_TURNS)
    for turn in conversation_history[start:]:
        messages.append({"role": "user",      "content": turn.user_message})
        messages.append({"role": "assistant",  "content": turn.assistant_message})
    messages.append({"role": "user", "content": user_text})

    full_reply = ""
    buffer = ""

    try:
        with requests.post(
            "https://llm.alem.ai/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            json={"model": "alemllm", "messages": messages, "stream": True},
            stream=True,
            timeout=30
        ) as resp:
            if resp.status_code != 200:
                print(f"LLM error: {resp.status_code}")
                sentence_queue.put(None)
                return ""

            for raw_line in resp.iter_lines():
                if not g_running:
                    break
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    break
                try:
                    token = json.loads(data)['choices'][0]['delta'].get('content', '')
                except (json.JSONDecodeError, KeyError):
                    continue

                if not token:
                    continue

                buffer     += token
                full_reply += token
                print(token, end='', flush=True)

                # Flush buffer at sentence boundary once we have enough text
                flush_at = -1
                for i, ch in enumerate(buffer):
                    if ch in SENTENCE_DELIMITERS and i >= MIN_CHUNK_LEN - 1:
                        flush_at = i

                if flush_at >= 0:
                    chunk = buffer[:flush_at + 1].strip()
                    buffer = buffer[flush_at + 1:]
                    if chunk:
                        sentence_queue.put(chunk)

    except Exception as e:
        print(f"\nLLM error: {e}")

    # Flush any trailing text
    if buffer.strip() and g_running:
        sentence_queue.put(buffer.strip())

    sentence_queue.put(None)  # sentinel
    return full_reply

# ============================================================================
# TEXT-TO-SPEECH — Azure Neural TTS (REST API)
# ============================================================================
def azure_tts(text: str) -> Optional[bytes]:
    """Returns raw WAV bytes (riff-16khz-16bit-mono-pcm) or None on error."""
    if not g_running or not text.strip():
        return None

    api_key = os.getenv("AZURE_TTS_API_KEY")
    if not api_key:
        print("ERROR: AZURE_TTS_API_KEY not set")
        return None

    ssml = (
        f'<speak version="1.0" xml:lang="kk-KZ">'
        f'<voice name="{AZURE_TTS_VOICE}">{html.escape(text)}</voice>'
        f'</speak>'
    )
    try:
        resp = requests.post(
            f"https://{AZURE_TTS_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
            headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
                "User-Agent": "Temirbek",
            },
            data=ssml.encode('utf-8'),
            timeout=10
        )
        if resp.status_code == 200:
            return resp.content  # includes WAV header
        print(f"TTS error: {resp.status_code} {resp.text[:120]}")
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def play_wav(wav_bytes: bytes, robot: 'G1AudioClient'):
    """Strip WAV header and stream raw PCM to the built-in speaker via DDS."""
    if not wav_bytes or not g_running:
        return
    # Azure TTS returns a standard 44-byte RIFF/WAV header before PCM data
    robot.play_pcm(wav_bytes[44:])

# ============================================================================
# TTS WORKER — synthesizes and plays sentences as they arrive from the queue
# ============================================================================
def tts_worker(sentence_queue: queue.Queue, robot: G1AudioClient):
    """
    Runs in its own thread.
    For each sentence from the queue: synthesize with Azure TTS, then play.
    Exits when it receives the None sentinel.
    """
    while g_running:
        try:
            sentence = sentence_queue.get(timeout=0.3)
        except queue.Empty:
            continue

        if sentence is None:
            break

        wav = azure_tts(sentence)
        if wav and g_running:
            robot.led(0, 100, 255)  # blue = speaking
            play_wav(wav, robot)

    robot.led(0, 0, 0)

# ============================================================================
# AUDIO RECORDING
# ============================================================================
def record_audio() -> List[int]:
    global g_is_recording, g_stop_recording

    pcm = []
    try:
        proc = subprocess.Popen(
            ['arecord', '-D', ALSA_INPUT_DEVICE, '-f', 'S16_LE',
             '-r', str(SAMPLE_RATE), '-c', '1', '-q'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        print("Recording... (Press ENTER to stop)")

        while g_running and g_is_recording and not g_stop_recording:
            chunk = proc.stdout.read(2048)
            if not chunk:
                break
            pcm.extend(struct.unpack(f'{len(chunk)//2}h', chunk))

        proc.terminate()
        proc.wait(timeout=1)
        print(f"Stopped ({len(pcm) / SAMPLE_RATE:.1f}s)")
    except Exception as e:
        print(f"Recording error: {e}")

    return pcm

# ============================================================================
# INPUT MONITOR THREAD
# ============================================================================
def input_monitor():
    global g_stop_recording
    while g_running:
        if g_is_recording:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                g_stop_recording = True
        time.sleep(0.05)

# ============================================================================
# MAIN
# ============================================================================
def main():
    global g_running, g_is_recording, g_stop_recording, conversation_history

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <NetworkInterface>")
        return 1

    for key in ("ALEMAI_STT_API_KEY", "ALEMAI_LLM_API_KEY", "AZURE_TTS_API_KEY"):
        if not os.getenv(key):
            print(f"ERROR: {key} not set")
            return 1

    robot = G1AudioClient(sys.argv[1])

    print("\n" + "=" * 44)
    print("  Temirbek — Conversation Buddy")
    print("=" * 44)
    print("  STT : AlemAI Whisper (Kazakh)")
    print("  LLM : AlemLLM (streaming SSE)")
    print("  TTS : Azure Neural — kk-KZ-DauletNeural")
    print(f"  Mic : {ALSA_INPUT_DEVICE} (USB)")
    print(f"  Spk : G1 built-in (DDS PlayStream)")
    print(f"  Mem : last {MAX_HISTORY_TURNS} turns")
    print("=" * 44)
    print("  ENTER       → start / stop recording")
    print("  'тазала'    → clear conversation memory")
    print("  Ctrl+C      → quit")
    print("=" * 44 + "\n")

    robot.led(0, 255, 0)
    time.sleep(0.4)
    robot.led(0, 0, 0)
    print("Ready.\n")

    threading.Thread(target=input_monitor, daemon=True).start()

    while g_running:
        print("Press ENTER to speak...")
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not g_running:
            break
        if not ready:
            continue

        sys.stdin.readline()
        if not g_running:
            break

        # --- Record ---
        g_is_recording  = True
        g_stop_recording = False
        robot.led(255, 0, 0)  # red = recording
        audio = record_audio()
        g_is_recording = False
        robot.led(0, 0, 0)

        if not g_running or not audio:
            break
        if len(audio) < SAMPLE_RATE * 0.5:
            print("Too short, skipping.\n")
            continue

        # --- STT ---
        print("Transcribing...")
        text = transcribe_audio(audio)
        if not text:
            print("Could not transcribe, try again.\n")
            continue

        print(f"\nYou: {text}")

        if any(w in text.lower() for w in ('тазала', 'жаңарт', 'clear')):
            conversation_history.clear()
            print("Memory cleared.\n")
            continue

        # --- LLM streaming + TTS pipeline ---
        print("Temirbek: ", end='', flush=True)
        robot.led(255, 165, 0)  # orange = thinking

        sq = queue.Queue()

        # TTS worker: synthesizes and plays each sentence as it arrives
        worker = threading.Thread(target=tts_worker, args=(sq, robot), daemon=True)
        worker.start()

        # LLM streams into sq; returns full reply when done
        full_reply = stream_llm(text, sq)

        # Wait until the last sentence finishes playing
        worker.join()
        print()

        if full_reply:
            conversation_history.append(
                ConversationTurn(user_message=text, assistant_message=full_reply)
            )
            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history.pop(0)
            print(f"[memory: {len(conversation_history)}/{MAX_HISTORY_TURNS} turns]\n")

    print("Goodbye.")
    robot.led(0, 0, 0)
    return 0

if __name__ == "__main__":
    sys.exit(main())
