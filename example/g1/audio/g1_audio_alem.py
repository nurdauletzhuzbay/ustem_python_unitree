#!/usr/bin/env python3
"""
Temirbek Voice Assistant
USB mic (VAD) + G1 built-in speaker (DDS PlayStream) + Azure Neural TTS + AlemAI STT/LLM

Pipeline: VAD record → STT → LLM (streaming) → Azure TTS per sentence → PlayStream
TTS synthesis of sentence N overlaps with LLM generation of sentence N+1.
"""

import os
import sys
import signal
import subprocess
import threading
import time
import struct
import wave
import io
import queue
import json
import html
import math
import collections
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import requests

# Ensure repo root is on the path so unitree_sdk2py is importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

AZURE_TTS_VOICE  = "kk-KZ-DauletNeural"            # Male Kazakh neural voice
AZURE_TTS_REGION = "eastus"

SAMPLE_RATE       = 16000
MAX_HISTORY_TURNS = 8
CHUNK_SIZE        = 1024

# VAD settings (calibrated for USB mic with motor noise)
SILENCE_THRESHOLD      = 3200
SILENCE_DURATION       = 1.5   # seconds of silence before stopping
MIN_SPEECH_DURATION    = 0.8   # seconds
SPEECH_START_THRESHOLD = 3500  # above noise peak to avoid false triggers


# ============================================================================
# GLOBAL STATE
# ============================================================================
g_running    = True
g_is_speaking = False

http_session = requests.Session()

@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str

conversation_history: List[ConversationTurn] = []
recent_transcriptions: collections.deque = collections.deque(maxlen=5)

HALLUCINATION_PHRASES = [
    "қасындастық елігі қазақстан",
    "қазақстан",
    "елігі",
    "субтитры",
    "переклад",
]

# ============================================================================
# SIGNAL HANDLER
# ============================================================================
def signal_handler(sig, frame):
    global g_running
    g_running = False
    print("\nShutting down...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ============================================================================
# G1 AUDIO CLIENT — LED + built-in speaker via DDS
# ============================================================================
class G1AudioClient:
    def __init__(self, network_interface: str):
        self.available = False
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            ChannelFactoryInitialize(0, network_interface)
            self.client = AudioClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
            self.client.SetVolume(100)
            self.available = True
            print(f"G1 SDK initialized on {network_interface}")
        except ImportError:
            print("Unitree SDK not available. Speaker + LED disabled.")
        except Exception as e:
            print(f"SDK init error: {e}")

    def led(self, r: int, g: int, b: int):
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception as e:
                print(f"LED error: {e}")

    def stream_pcm_realtime(self, pcm_queue: queue.Queue,
                             app_name: str = "temirbek",
                             stream_id: str = None):
        """
        Drain pcm_queue and stream all PCM chunks to the speaker at real-time pace.
        Uses a single stream_id for the whole response so there are no gaps between
        phrases. Paces sends by wall-clock elapsed time instead of per-chunk sleeps.

        pcm_queue items: bytes (raw PCM) or None (sentinel = done).
        """
        if not self.available:
            # Drain queue so callers don't block
            while True:
                try:
                    item = pcm_queue.get(timeout=0.3)
                    if item is None:
                        break
                except queue.Empty:
                    continue
            return

        if stream_id is None:
            stream_id = str(int(time.time() * 1000))

        BYTES_PER_SEC = 16000 * 2          # 32000 bytes/s at 16kHz 16-bit mono
        CHUNK = 3200                        # 100 ms per send — small enough to stay ahead
        total_sent = 0
        t_start    = None

        try:
            pending = b''
            done    = False

            while not done or pending:
                # Fill pending buffer from queue without blocking mid-stream
                while not done and len(pending) < CHUNK * 4:
                    try:
                        item = pcm_queue.get(timeout=0.05)
                        if item is None:
                            done = True
                        else:
                            pending += item
                    except queue.Empty:
                        break

                if not pending:
                    if done:
                        break
                    continue

                chunk = pending[:CHUNK]
                pending = pending[CHUNK:]

                if t_start is None:
                    t_start = time.time()

                ret, _ = self.client.PlayStream(app_name, stream_id, chunk)
                if ret != 0:
                    print(f"  PlayStream error: code {ret}")
                    break

                total_sent += len(chunk)

                # Sleep until the robot's playback cursor catches up to what we sent
                # (minus a small lead-time so the buffer never starves)
                lead = 0.15  # keep 150 ms ahead of playback cursor
                target = t_start + total_sent / BYTES_PER_SEC - lead
                now = time.time()
                if target > now:
                    time.sleep(target - now)

            # Wait for the last bytes to finish playing
            if t_start is not None:
                end_time = t_start + total_sent / BYTES_PER_SEC
                remaining = end_time - time.time()
                if remaining > 0:
                    time.sleep(remaining)

            self.client.PlayStop(app_name)

        except Exception as e:
            print(f"  PlayStream error: {e}")

    def test_speaker(self):
        """Play a short beep to verify the speaker is working."""
        if not self.available:
            return
        print("Testing speaker...")
        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
                   for i in range(int(SAMPLE_RATE * 0.3))]
        pcm = struct.pack(f'{len(samples)}h', *samples)
        q = queue.Queue()
        q.put(pcm)
        q.put(None)
        self.stream_pcm_realtime(q)

# ============================================================================
# AUDIO UTILITIES
# ============================================================================
def calculate_rms(audio_chunk: bytes) -> float:
    try:
        count = len(audio_chunk) // 2
        shorts = struct.unpack(f'{count}h', audio_chunk)
        return (sum(s * s for s in shorts) / count) ** 0.5
    except:
        return 0.0

def build_wav_bytes(audio_bytes: bytes) -> bytes:
    """Build a WAV file in memory from raw PCM bytes."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_bytes)
    return buf.getvalue()

# ============================================================================
# RECORDING WITH VAD (USB mic via arecord)
# ============================================================================
def record_with_vad() -> Optional[bytes]:
    """Record from USB mic, auto-stop on silence. Returns raw PCM bytes."""
    if g_is_speaking:
        return None

    cmd = ['arecord', '-D', ALSA_INPUT_DEVICE,
           '-f', 'S16_LE', '-r', str(SAMPLE_RATE), '-c', '1', '-q']
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=CHUNK_SIZE * 2)

        speech_chunks   = []
        silence_frames  = 0
        speech_started  = False
        bytes_per_check = int(SAMPLE_RATE * 0.1) * 2  # 100ms
        buf             = b''
        rms_history     = []
        total_bytes     = 0

        print("Listening...")

        while g_running and not g_is_speaking:
            chunk = proc.stdout.read(CHUNK_SIZE * 2)
            if not chunk:
                break
            buf += chunk

            if len(buf) >= bytes_per_check:
                check = buf[:bytes_per_check]
                buf   = buf[bytes_per_check:]
                rms   = calculate_rms(check)

                if not speech_started and rms > SPEECH_START_THRESHOLD:
                    speech_started = True
                    silence_frames = 0
                    print("Speech detected...")

                if speech_started:
                    speech_chunks.append(check)
                    total_bytes += len(check)
                    rms_history.append(rms)

                    if rms < SILENCE_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    if silence_frames * 0.1 >= SILENCE_DURATION:
                        print("Silence, processing...")
                        break

        proc.terminate()
        proc.wait(timeout=2)

        if speech_chunks and rms_history:
            duration = (total_bytes // 2) / SAMPLE_RATE
            if duration < MIN_SPEECH_DURATION:
                print(f"Too short ({duration:.1f}s), skipping.")
                return None
            print(f"Recorded {duration:.1f}s, avg RMS: {sum(rms_history)/len(rms_history):.0f}")
            return b''.join(speech_chunks)

        return None
    except Exception as e:
        print(f"Recording error: {e}")
        return None

# ============================================================================
# SPEECH-TO-TEXT — AlemAI Whisper
# ============================================================================
def is_hallucination(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True
    t = text.lower().strip()
    for phrase in HALLUCINATION_PHRASES:
        if phrase.lower() in t:
            return True
    if sum(1 for x in recent_transcriptions if x == t) >= 2:
        return True
    recent_transcriptions.append(t)
    return False

def transcribe_audio(audio_bytes: bytes) -> str:
    if not g_running:
        return ""
    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return ""
    try:
        wav_data = build_wav_bytes(audio_bytes)
        resp = http_session.post(
            "https://llm.alem.ai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={'file': ('audio.wav', io.BytesIO(wav_data), 'audio/wav')},
            data={'model': 'speech-to-text-kk'},
            timeout=15
        )
        if resp.status_code == 200:
            text = resp.json().get('text', '').strip()
            if is_hallucination(text):
                print(f"Hallucination filtered: '{text}'")
                return ""
            return text
        print(f"STT error: {resp.status_code}")
        return ""
    except Exception as e:
        print(f"STT error: {e}")
        return ""

# ============================================================================
# LLM — AlemAI (streams tokens, flushes phrases to TTS queue immediately)
# ============================================================================
SYSTEM_PROMPT = (
    "Сен Темірбек деген Unitree G1 роботысың. Қазақша сөйлейсің.\n"
    "Мақсатың — адаммен шынайы, жылы әңгіме жүргізу. "
    "Жауап берген соң, өзің де қызықты сұрақ қой немесе тақырыпты дамыт — "
    "бос тұрма, диалогты ұстап тұр. "
    "Жауаптарың қысқа (1–3 сөйлем), табиғи, дос сияқты. Ресми тіл қолданба."
)

# Split on sentence-ending punctuation immediately; soft-split on comma/colon
# only after enough chars to avoid tiny TTS calls.
HARD_SPLIT = frozenset('.!?…')
SOFT_SPLIT = frozenset(',;:')
MIN_CHARS  = 60

def get_llm_response(user_text: str, phrase_queue: queue.Queue) -> str:
    """
    Stream LLM tokens to the screen.
    Push complete phrases into phrase_queue as they form (None sentinel at end).
    Returns the full reply.
    """
    if not g_running:
        phrase_queue.put(None)
        return ""

    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        phrase_queue.put(None)
        return ""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    start = max(0, len(conversation_history) - MAX_HISTORY_TURNS)
    for turn in conversation_history[start:]:
        messages.append({"role": "user",      "content": turn.user_message})
        messages.append({"role": "assistant",  "content": turn.assistant_message})
    messages.append({"role": "user", "content": user_text})

    full_reply = ""
    buf = ""

    try:
        with http_session.post(
            "https://llm.alem.ai/v1/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"},
            json={"model": "alemllm", "messages": messages, "stream": True},
            stream=True,
            timeout=30
        ) as resp:
            if resp.status_code != 200:
                print(f"LLM error: {resp.status_code}")
                phrase_queue.put(None)
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

                buf        += token
                full_reply += token
                print(token, end='', flush=True)

                # Find the last flush point in the buffer
                cut = -1
                for i, ch in enumerate(buf):
                    if ch in HARD_SPLIT:
                        cut = i
                    elif ch in SOFT_SPLIT and len(buf) >= MIN_CHARS:
                        cut = i

                if cut >= 0:
                    phrase = buf[:cut + 1].strip()
                    buf = buf[cut + 1:]
                    if phrase:
                        phrase_queue.put(phrase)

    except Exception as e:
        print(f"\nLLM error: {e}")

    if buf.strip():
        phrase_queue.put(buf.strip())

    phrase_queue.put(None)
    return full_reply


# ============================================================================
# TTS + PLAYBACK PIPELINE
#
#  phrase_queue → [tts_fetcher] → pcm_queue → [stream_pcm_realtime]
#
# tts_fetcher calls Azure TTS for each phrase and puts raw PCM into pcm_queue.
# stream_pcm_realtime drains pcm_queue using a single stream_id with clock-
# paced sends, so there are zero gaps between phrases.
# ============================================================================
def speak_response(phrase_queue: queue.Queue, robot: G1AudioClient):
    global g_is_speaking

    pcm_queue  = queue.Queue(maxsize=4)
    stream_id  = str(int(time.time() * 1000))

    def tts_fetcher():
        while g_running:
            try:
                phrase = phrase_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if phrase is None:
                pcm_queue.put(None)
                break
            wav = azure_tts(phrase)
            # Strip 44-byte WAV header → raw PCM, or None on error
            pcm_queue.put(wav[44:] if wav else b'')

    fetcher = threading.Thread(target=tts_fetcher, daemon=True)
    fetcher.start()

    robot.led(0, 100, 255)
    g_is_speaking = True
    robot.stream_pcm_realtime(pcm_queue, stream_id=stream_id)
    g_is_speaking = False
    robot.led(0, 0, 0)

    fetcher.join()


# ============================================================================
# TEXT-TO-SPEECH — Azure Neural TTS (REST API)
# ============================================================================
def azure_tts(text: str) -> Optional[bytes]:
    """Returns WAV bytes (riff-16khz-16bit-mono-pcm) or None on error."""
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
        resp = http_session.post(
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
            return resp.content  # includes 44-byte WAV header
        print(f"TTS error: {resp.status_code} {resp.text[:120]}")
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ============================================================================
# MAIN
# ============================================================================
def main():
    global g_running

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <NetworkInterface> [--test-speaker]")
        return 1

    for key in ("ALEMAI_STT_API_KEY", "ALEMAI_LLM_API_KEY", "AZURE_TTS_API_KEY"):
        if not os.getenv(key):
            print(f"ERROR: {key} not set")
            return 1

    robot = G1AudioClient(sys.argv[1])

    if '--test-speaker' in sys.argv:
        robot.test_speaker()
        return 0

    print("\n" + "=" * 46)
    print("  Temirbek — Conversation Buddy")
    print("=" * 46)
    print("  STT : AlemAI Whisper (Kazakh)")
    print("  LLM : AlemLLM (streaming SSE)")
    print("  TTS : Azure Neural — kk-KZ-DauletNeural")
    print(f"  Mic : {ALSA_INPUT_DEVICE} (USB)")
    print(f"  Spk : G1 built-in (DDS PlayStream)")
    print(f"  VAD : silence={SILENCE_THRESHOLD} speech={SPEECH_START_THRESHOLD}")
    print(f"  Mem : last {MAX_HISTORY_TURNS} turns")
    print("=" * 46)
    print("  Say 'тазала/clear' → reset memory")
    print("  Say 'тоқта/stop'   → quit")
    print("  Ctrl+C             → quit")
    print("=" * 46 + "\n")

    robot.led(0, 255, 0)
    time.sleep(0.4)
    robot.led(0, 0, 0)
    print("Ready.\n")

    try:
        while g_running:
            robot.led(0, 255, 0)  # green = listening
            audio = record_with_vad()
            robot.led(0, 0, 0)

            if not g_running or not audio:
                continue

            # STT
            robot.led(255, 255, 0)  # yellow = transcribing
            print("Transcribing...")
            text = transcribe_audio(audio)
            if not g_running or not text:
                print("Transcription failed.\n")
                continue

            print(f"\nYou: {text}")

            lower = text.lower()
            if any(w in lower for w in ('тазала', 'жаңарт', 'clear', 'reset')):
                conversation_history.clear()
                print("Memory cleared.\n")
                continue
            if any(w in lower for w in ('тоқта', 'stop', 'exit', 'quit', 'шығу')):
                print("Stopping.")
                break

            # LLM streams phrases → TTS fetcher → real-time playback (all parallel)
            print("Temirbek: ", end='', flush=True)
            robot.led(255, 165, 0)  # orange = thinking

            phrase_q = queue.Queue()

            # speak_response runs in a thread: fetches TTS and plays in real-time
            speaker = threading.Thread(
                target=speak_response, args=(phrase_q, robot), daemon=True
            )
            speaker.start()

            # LLM streams tokens to screen and pushes phrases into phrase_q
            full_reply = get_llm_response(text, phrase_q)
            speaker.join()
            print()

            if not full_reply:
                continue

            conversation_history.append(
                ConversationTurn(user_message=text, assistant_message=full_reply)
            )
            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history.pop(0)
            print(f"[memory: {len(conversation_history)}/{MAX_HISTORY_TURNS} turns]\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        g_running = False
        http_session.close()
        robot.led(0, 0, 0)

    print("Goodbye.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
