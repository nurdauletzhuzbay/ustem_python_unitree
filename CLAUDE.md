# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python SDK for controlling Unitree humanoid robots (primarily the G1 model), forked from the official `unitree_sdk2_python`. The project has been extended with a Kazakh-language voice assistant ("Temirbek") that runs on the G1 robot, using AlemAI for STT/LLM and Piper or Edge TTS for speech synthesis.

## Installation

```bash
pip3 install -e .
```

If CycloneDDS is not found:
```bash
# Build CycloneDDS from source
cd ~ && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install

# Then install the SDK
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```

Dependencies: `cyclonedds==0.10.2`, `numpy`, `opencv-python`, `pyaudio`, `requests`

## Running Examples

All examples take a network interface name as argument (e.g., `eth0`, `enp2s0`):

```bash
# G1 high-level locomotion control
python3 ./example/g1/high_level/g1_loco_client_example.py <interface>

# G1 low-level motor control (disable sport_mode first via app)
python3 ./example/g1/low_level/g1_low_level_example.py <interface>

# Voice assistant (requires ALEMAI_STT_API_KEY and ALEMAI_LLM_API_KEY env vars)
python3 ./example/g1/audio/g1_audio_alem.py <interface>
python3 ./example/g1/audio/g1_audio_alem_streaming.py <interface>

# Basic DDS pub/sub test
python3 ./example/helloworld/publisher.py
python3 ./example/helloworld/subscriber.py
```

## Architecture

### Communication Stack

```
Robot <--DDS (CycloneDDS v0.10.2)--> ChannelFactory (Singleton)
                                           |
                              +-----------+-----------+
                              |                       |
                    Pub/Sub Channels              RPC Clients
                    (ChannelPublisher,            (ClientBase →
                     ChannelSubscriber)            specific clients)
```

**`unitree_sdk2py/core/channel.py`** — The foundation. `ChannelFactory` is a singleton initialized once with a domain ID and optional network interface. All publishers and subscribers share this factory. DDS topics follow the naming convention in `core/channel_name.py`: `rt/api/{serviceName}/request` and `rt/api/{serviceName}/response`.

**`unitree_sdk2py/rpc/`** — RPC framework built on top of DDS pub/sub. `ClientBase` sends requests and waits on futures with timeouts. `ServerBase` registers handlers and dispatches responses. Lease-based access control (`lease_client.py`/`lease_server.py`) enables exclusive robot control.

**`unitree_sdk2py/idl/`** — Message definitions (IDL). Two robot-specific namespaces:
- `unitree_go/` — Go2/B2 quadruped messages (12 DOF joints, `LowState_`, `LowCmd_`, `SportModeState_`)
- `unitree_hg/` — G1/H1-2 humanoid messages (29 DOF joints, hand commands, pressure sensors, `MainBoardState_`)
- `idl/default.py` — Convenience aliases for all message types

**`unitree_sdk2py/comm/motion_switcher/`** — Switches the robot between high-level sport mode and low-level direct control. Must be used before sending low-level motor commands.

### Initialization Pattern

Every script must initialize `ChannelFactory` before creating any publishers, subscribers, or clients:

```python
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
ChannelFactoryInitialize(domain_id=0, networkInterface="eth0")
# or for auto-detect: ChannelFactoryInitialize(0)
```

### Voice Assistant (G1-specific)

`example/g1/audio/` contains the Temirbek voice assistant variants:
- **`g1_audio_alem.py`** — Push-to-talk (ENTER key), uses Piper TTS (local, `/opt/piper/`)
- **`g1_audio_alem_streaming.py`** — Continuous listening, streaming Edge TTS (`kk-KZ-DauletNeural`)
- **`g1_audio_alem_continuous.py`** — Continuous listening variant

Audio pipeline: USB mic → `arecord` → AlemAI Whisper STT → AlemAI LLM → Edge TTS / Piper → `aplay` → USB speaker

Required environment variables:
- `ALEMAI_STT_API_KEY` — AlemAI speech-to-text API key
- `ALEMAI_LLM_API_KEY` — AlemAI LLM API key
- `AZURE_TTS_API_KEY` — Azure Cognitive Services TTS key (region: `eastus`, voice: `kk-KZ-DauletNeural`)

Hardware config (ALSA device names):
- Input: `plughw:CARD=Device,DEV=0` (JMTek USB microphone)
- Output: `plughw:CARD=Audio,DEV=0` (Moshi USB speaker)

The LLM response is streamed via SSE and split into sentences; each sentence is synthesized by Azure TTS and played immediately while the LLM continues generating the next sentence (producer-consumer pipeline via a queue + worker thread).

### G1 Loco Client

The `LocoClient` (imported from `unitree_sdk2py.g1.loco.g1_loco_client`) provides high-level locomotion commands: Damp, StandUp, Squat, Move (forward/lateral/rotate), WaveHand, ShakeHand, etc. This module is not in the repo source — it's expected to come from the installed Unitree C++ SDK bindings or a separate package.

### Low-Level Motor Control

For direct joint control on G1, use `unitree_hg` message types (`LowCmd_`, `LowState_`). Always:
1. Switch motion mode via `MotionSwitcherClient` first
2. Use `crc.crc32()` from `unitree_sdk2py/utils/crc.py` to compute command checksums
3. Send at 500Hz control loop frequency
