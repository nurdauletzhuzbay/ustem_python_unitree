#!/usr/bin/env python3
"""
Temirbek Voice Assistant - Continuous Listening Version
Automatically detects speech and responds continuously
External microphone (USB) + External speaker (USB) + Edge TTS + AlemAI STT/LLM
"""

import os
import sys
import signal
import subprocess
import threading
import time
import struct
import wave
import tempfile
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
import requests
import collections
import audioop

# Edge TTS (pip install edge-tts)
try:
    import edge_tts
except Exception:
    edge_tts = None

# ============================================================================
# CONFIGURATION
# ============================================================================
ALSA_OUTPUT_DEVICE = "plughw:CARD=Audio,DEV=0"  # External speaker (Moshi)
ALSA_INPUT_DEVICE = "plughw:CARD=Device,DEV=0"  # External microphone (JMTek)

# Edge TTS voice (male Kazakh)
EDGE_TTS_VOICE = "kk-KZ-DauletNeural"

SAMPLE_RATE = 16000
MAX_HISTORY_TURNS = 5
CHUNK_SIZE = 1024

# Voice Activity Detection settings
SILENCE_THRESHOLD = 1500  # RMS threshold for silence (very high for motor noise)
SILENCE_DURATION = 2.0   # Seconds of silence before stopping (increased)
MIN_SPEECH_DURATION = 1.5  # Minimum speech duration to process (increased)
SPEECH_START_THRESHOLD = 2200  # RMS threshold to start recording (very high for motor noise)

# ============================================================================
# GLOBAL STATE
# ============================================================================
g_running = True
g_is_listening = True
g_is_speaking = False

@dataclass
class ConversationTurn:
    user_message: str
    assistant_message: str

conversation_history: List[ConversationTurn] = []
recent_transcriptions: collections.deque = collections.deque(maxlen=5)  # Track last 5 transcriptions

# Known Whisper hallucinations for Kazakh
HALLUCINATION_PHRASES = [
    "қасындастық елігі қазақстан",
    "қазақстан",
    "елігі",
    "субтитры",
    "переклад",
    # Add more as you discover them
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
# UNITREE G1 AUDIO CLIENT (LED CONTROL)
# ============================================================================
class G1AudioClient:
    """Simple wrapper for G1 LED control via DDS"""

    def __init__(self, network_interface: str):
        self.network_interface = network_interface
        try:
            from unitree.robot.channel.channel_factory import ChannelFactory
            from unitree.robot.g1.audio.g1_audio_client import AudioClient

            ChannelFactory.Instance().Init(0, network_interface)
            self.client = AudioClient()
            self.client.Init()
            self.client.SetTimeout(10.0)
            self.available = True
        except ImportError:
            print("Warning: Unitree SDK not available. LED control disabled.")
            self.available = False

    def led_control(self, r: int, g: int, b: int):
        """Control G1 head LED"""
        if self.available:
            try:
                self.client.LedControl(r, g, b)
            except Exception as e:
                print(f"LED control error: {e}")

# ============================================================================
# AUDIO UTILITIES
# ============================================================================
def calibrate_microphone(duration: int = 5):
    """Calibrate microphone by measuring background noise levels"""
    print(f"\n🎤 Calibrating microphone for {duration} seconds...")
    print("Please keep the environment as it normally is (with motors running if applicable)")

    cmd = [
        'arecord',
        '-D', ALSA_INPUT_DEVICE,
        '-f', 'S16_LE',
        '-r', str(SAMPLE_RATE),
        '-c', '1',
        '-q'
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=CHUNK_SIZE * 2
    )

    rms_values = []
    start_time = time.time()

    while time.time() - start_time < duration:
        chunk = process.stdout.read(CHUNK_SIZE * 2)
        if chunk:
            rms = calculate_rms(chunk)
            rms_values.append(rms)

    process.terminate()
    process.wait(timeout=1)

    if rms_values:
        avg_rms = sum(rms_values) / len(rms_values)
        max_rms = max(rms_values)
        min_rms = min(rms_values)

        print(f"\n📊 Calibration Results:")
        print(f"  Average noise level: {avg_rms:.0f}")
        print(f"  Max noise level: {max_rms:.0f}")
        print(f"  Min noise level: {min_rms:.0f}")
        print(f"\n💡 Recommended settings:")
        print(f"  SILENCE_THRESHOLD = {int(max_rms * 1.5)}")
        print(f"  SPEECH_START_THRESHOLD = {int(max_rms * 2.5)}")
        print(f"\nNow speak loudly for 3 seconds...")

        # Measure speech
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=CHUNK_SIZE * 2
        )

        speech_values = []
        start_time = time.time()

        while time.time() - start_time < 3:
            chunk = process.stdout.read(CHUNK_SIZE * 2)
            if chunk:
                rms = calculate_rms(chunk)
                speech_values.append(rms)

        process.terminate()
        process.wait(timeout=1)

        if speech_values:
            avg_speech = sum(speech_values) / len(speech_values)
            max_speech = max(speech_values)

            print(f"\n🗣️  Speech Levels:")
            print(f"  Average: {avg_speech:.0f}")
            print(f"  Peak: {max_speech:.0f}")
            print(f"\n✅ Final Recommended Settings:")
            print(f"  SILENCE_THRESHOLD = {int(max_rms * 1.5)}")
            print(f"  SPEECH_START_THRESHOLD = {int(avg_speech * 0.7)}")
            print()

def calculate_rms(audio_chunk: bytes) -> float:
    """Calculate RMS (Root Mean Square) of audio chunk for VAD"""
    try:
        count = len(audio_chunk) // 2
        format_str = f'{count}h'
        shorts = struct.unpack(format_str, audio_chunk)
        sum_squares = sum(s**2 for s in shorts)
        rms = (sum_squares / count) ** 0.5
        return rms
    except:
        return 0

def create_wav_file(audio_pcm: List[int], filename: str):
    """Create WAV file from PCM data"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)

        audio_bytes = struct.pack(f'{len(audio_pcm)}h', *audio_pcm)
        wav_file.writeframes(audio_bytes)

# ============================================================================
# SPEECH-TO-TEXT (AlemAI Whisper)
# ============================================================================
def transcribe_audio(audio_pcm: List[int]) -> str:
    """Transcribe audio using AlemAI Whisper"""
    if not g_running:
        return ""

    api_key = os.getenv("ALEMAI_STT_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return ""

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        wav_file = tmp_file.name

    try:
        create_wav_file(audio_pcm, wav_file)

        url = "https://llm.alem.ai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {api_key}"}

        with open(wav_file, 'rb') as f:
            files = {'file': ('audio.wav', f, 'audio/wav')}
            data = {'model': 'speech-to-text-kk'}

            response = requests.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=15
            )

        if response.status_code == 200 and g_running:
            result = response.json()
            transcription = result.get('text', '').strip()

            if is_hallucination(transcription):
                print(f"⚠️  Detected hallucination: '{transcription}' - ignoring")
                return ""

            return transcription
        else:
            print(f"STT error: {response.status_code}")
            return ""

    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)

def is_hallucination(text: str) -> bool:
    """Check if transcription is likely a Whisper hallucination"""
    if not text:
        return True

    text_lower = text.lower().strip()

    for phrase in HALLUCINATION_PHRASES:
        if phrase.lower() in text_lower:
            return True

    if text_lower in recent_transcriptions:
        count = sum(1 for t in recent_transcriptions if t == text_lower)
        if count >= 2:
            return True

    recent_transcriptions.append(text_lower)

    if len(text_lower) < 3:
        return True

    return False

# ============================================================================
# LLM (AlemAI)
# ============================================================================
def get_llm_response(user_text: str) -> str:
    """Get response from AlemAI LLM"""
    if not g_running:
        return ""

    api_key = os.getenv("ALEMAI_LLM_API_KEY")
    if not api_key:
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        return ""

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Temirbek, a helpful robot assistant. User speaks Kazakh. "
                    "Reply in Kazakh briefly (1-2 sentences). "
                    "Remember the conversation context."
                )
            }
        ]

        start_idx = max(0, len(conversation_history) - MAX_HISTORY_TURNS)
        for turn in conversation_history[start_idx:]:
            messages.append({"role": "user", "content": turn.user_message})
            messages.append({"role": "assistant", "content": turn.assistant_message})

        messages.append({"role": "user", "content": user_text})

        url = "https://llm.alem.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {"model": "alemllm", "messages": messages}

        response = requests.post(url, headers=headers, json=payload, timeout=10)

        if response.status_code == 200 and g_running:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            print(f"LLM error: {response.status_code}")
            return ""

    except Exception as e:
        print(f"LLM error: {e}")
        return ""

# ============================================================================
# TEXT-TO-SPEECH (Edge TTS)
# ============================================================================
async def _edge_tts_to_wav(text: str, wav_path: str, voice: str):
    """
    Generate a WAV file using Microsoft Edge TTS.
    """
    if edge_tts is None:
        raise RuntimeError("edge-tts is not installed. Run: python3 -m pip install --user edge-tts")

    # You can tune rate/volume if you want:
    # rate="+0%" volume="+0%"
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(wav_path)

def generate_speech(text: str) -> Optional[List[int]]:
    """
    Generate speech using Edge TTS and return PCM int16 list at SAMPLE_RATE mono.
    """
    if not g_running:
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
            wav_file = tmp_wav.name

        with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as tmp_pcm:
            pcm_file = tmp_pcm.name

        # 1) Edge TTS -> WAV
        asyncio.run(_edge_tts_to_wav(text=text, wav_path=wav_file, voice=EDGE_TTS_VOICE))

        # 2) Convert WAV -> raw PCM 16kHz mono int16 (same format your aplay expects)
        cmd = f"ffmpeg -y -i {wav_file} -ar {SAMPLE_RATE} -ac 1 -f s16le {pcm_file}"
        subprocess.run(cmd, shell=True, check=True, capture_output=True)

        if g_running and os.path.exists(pcm_file):
            with open(pcm_file, 'rb') as f:
                pcm_data = f.read()

            audio = list(struct.unpack(f'{len(pcm_data)//2}h', pcm_data))
            return audio

        return None

    except Exception as e:
        print(f"TTS error: {e}")
        return None
    finally:
        for f in ["wav_file", "pcm_file"]:
            try:
                path = locals().get(f)
                if path and os.path.exists(path):
                    os.remove(path)
            except:
                pass

# ============================================================================
# CONTINUOUS AUDIO RECORDING WITH VAD
# ============================================================================
def record_with_vad() -> Optional[List[int]]:
    """Record audio continuously and detect speech using VAD"""
    global g_is_listening

    if g_is_speaking:
        return None

    cmd = [
        'arecord',
        '-D', ALSA_INPUT_DEVICE,
        '-f', 'S16_LE',
        '-r', str(SAMPLE_RATE),
        '-c', '1',
        '-q'
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=CHUNK_SIZE * 2
        )

        speech_frames = []
        silence_frames = 0
        speech_started = False
        frames_per_check = int(SAMPLE_RATE * 0.1)  # 100ms

        print("🎤 Listening... (speak naturally)")

        buffer = b''
        rms_history = []

        while g_running and g_is_listening and not g_is_speaking:
            chunk = process.stdout.read(CHUNK_SIZE * 2)
            if not chunk:
                break

            buffer += chunk

            if len(buffer) >= frames_per_check * 2:
                check_chunk = buffer[:frames_per_check * 2]
                buffer = buffer[frames_per_check * 2:]

                rms = calculate_rms(check_chunk)

                if not speech_started and rms > SPEECH_START_THRESHOLD:
                    speech_started = True
                    silence_frames = 0
                    print("🗣️  Speech detected...")

                if speech_started:
                    samples = struct.unpack(f'{len(check_chunk)//2}h', check_chunk)
                    speech_frames.extend(samples)
                    rms_history.append(rms)

                    if rms < SILENCE_THRESHOLD:
                        silence_frames += 1
                    else:
                        silence_frames = 0

                    silence_duration = silence_frames * 0.1
                    if silence_duration >= SILENCE_DURATION:
                        print("⏹️  Silence detected, processing...")
                        break

        process.terminate()
        process.wait(timeout=1)

        if speech_frames and rms_history:
            duration = len(speech_frames) / SAMPLE_RATE
            avg_rms = sum(rms_history) / len(rms_history)
            max_rms = max(rms_history)

            if duration < MIN_SPEECH_DURATION:
                print(f"⚠️  Speech too short ({duration:.1f}s), ignoring...")
                return None

            if avg_rms < SPEECH_START_THRESHOLD * 0.8:
                print(f"⚠️  Audio too quiet (avg: {avg_rms:.0f}), probably not speech...")
                return None

            if max_rms < avg_rms * 1.3:
                print(f"⚠️  Constant noise detected (no speech dynamics), ignoring...")
                return None

            print(f"✓ Valid speech detected ({duration:.1f}s, avg RMS: {avg_rms:.0f})")
            return speech_frames

        return None

    except Exception as e:
        print(f"Recording error: {e}")
        return None

# ============================================================================
# AUDIO PLAYBACK
# ============================================================================
def play_audio(audio_data: List[int]) -> bool:
    """Play audio through USB speaker"""
    global g_is_speaking

    if not audio_data or not g_running:
        return False

    g_is_speaking = True

    try:
        audio_bytes = struct.pack(f'{len(audio_data)}h', *audio_data)

        cmd = [
            'aplay',
            '-D', ALSA_OUTPUT_DEVICE,
            '-f', 'S16_LE',
            '-r', str(SAMPLE_RATE),
            '-c', '1',
            '-q'
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        process.communicate(input=audio_bytes)
        return True

    except Exception as e:
        print(f"Playback error: {e}")
        return False
    finally:
        g_is_speaking = False

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================
def process_conversation(audio_client: G1AudioClient):
    """Main conversation processing loop"""
    global conversation_history

    while g_running:
        audio_client.led_control(0, 255, 0)  # Green = listening
        audio = record_with_vad()
        audio_client.led_control(0, 0, 0)

        if not g_running:
            break

        if not audio:
            time.sleep(0.1)
            continue

        audio_client.led_control(255, 255, 0)  # Yellow = processing
        print("📝 Transcribing...")
        transcription = transcribe_audio(audio)

        if not g_running or not transcription:
            audio_client.led_control(0, 0, 0)
            print("❌ Transcription failed.")
            continue

        print(f"👤 You: {transcription}")

        lower_trans = transcription.lower()
        if any(word in lower_trans for word in ['clear', 'тазала', 'жаңарт', 'reset']):
            conversation_history.clear()
            audio_client.led_control(0, 0, 0)
            print("🗑️  Memory cleared.")
            continue

        if any(word in lower_trans for word in ['stop', 'тоқта', 'exit', 'quit', 'шығу']):
            print("👋 Stopping assistant...")
            audio_client.led_control(0, 0, 0)
            break

        print("🤔 Thinking...")
        reply = get_llm_response(transcription)

        if not g_running:
            break

        if reply:
            print(f"🤖 Temirbek: {reply}")

            turn = ConversationTurn(user_message=transcription, assistant_message=reply)
            conversation_history.append(turn)

            if len(conversation_history) > MAX_HISTORY_TURNS:
                conversation_history.pop(0)

            print(f"💾 Memory: {len(conversation_history)} turns")

            print("🎙️  Generating speech...")
            tts_audio = generate_speech(reply)

            if not g_running:
                break

            if tts_audio:
                audio_client.led_control(0, 100, 255)  # Blue = speaking
                print("🔊 Speaking...")
                play_audio(tts_audio)
                audio_client.led_control(0, 0, 0)

            print("✅ Done.\n")

        audio_client.led_control(0, 0, 0)
        time.sleep(0.5)

# ============================================================================
# MAIN
# ============================================================================
def main():
    global g_running

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [NetworkInterface] [--calibrate]")
        return 1

    if '--calibrate' in sys.argv or '-c' in sys.argv:
        calibrate_microphone(duration=5)
        return 0

    if not os.getenv("ALEMAI_STT_API_KEY"):
        print("ERROR: ALEMAI_STT_API_KEY not set")
        return 1

    if not os.getenv("ALEMAI_LLM_API_KEY"):
        print("ERROR: ALEMAI_LLM_API_KEY not set")
        return 1

    audio_client = G1AudioClient(sys.argv[1])

    print("\n" + "=" * 50)
    print("  Temirbek Voice Assistant - CONTINUOUS MODE")
    print("=" * 50)
    print("  STT: AlemAI (Kazakh)")
    print("  LLM: AlemLLM (Kazakh)")
    print(f"  TTS: Edge TTS - {EDGE_TTS_VOICE} (Male Kazakh)")
    print("  Mode: Continuous listening with auto-detection")
    print(f"  Memory: {MAX_HISTORY_TURNS} turns")
    print("=" * 50)
    print("Features:")
    print("  ✓ Automatic speech detection")
    print("  ✓ Hands-free operation")
    print("  ✓ Continuous conversation")
    print("\nVoice Commands:")
    print("  Say 'clear/тазала/жаңарт': Reset memory")
    print("  Say 'stop/тоқта/exit': Quit assistant")
    print("  Ctrl+C: Force quit")
    print("=" * 50 + "\n")

    print("⚙️  Adjusting microphone sensitivity...")
    print(f"  Silence threshold: {SILENCE_THRESHOLD}")
    print(f"  Speech threshold: {SPEECH_START_THRESHOLD}")
    print(f"  Silence duration: {SILENCE_DURATION}s")
    print("  (Adjust these in script if needed)\n")

    audio_client.led_control(0, 255, 0)
    time.sleep(1)
    audio_client.led_control(0, 0, 0)

    print("✅ Ready! Start speaking...\n")

    try:
        process_conversation(audio_client)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        g_running = False
        audio_client.led_control(0, 0, 0)
        print("\n👋 Goodbye!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
