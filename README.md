# VoiceBot — Real-Time Multi-Agent Voice Assistant

A production-ready voice assistant built with Python and LiveKit. Speak into your mic and the assistant responds in real time. Features a specialized Math Agent that independently handles mathematical queries, and a memory system that remembers users across sessions.

## What It Does

- Real-time voice conversations via microphone
- Multi-agent system — a Math Specialist Agent independently classifies and handles math queries
- RAG-based memory using Qdrant vector database — the assistant remembers things about you across sessions
- Conversation history stored in SQLite
- Session recording via LiveKit Egress
- Invite system — share a link for someone else to join your room

## Architecture

```
User (Browser) ──► LiveKit Server ──► Main Agent (agent.py)
                                           │
                              ┌────────────┴────────────┐
                              ▼                         ▼
                       Math Agent              Memory Service
                   (math_participant.py)     (memory_service.py)
                       Groq LLaMA 3.1          Qdrant + SQLite
```

- **Main Agent** — handles voice, STT, TTS, and general queries (OpenAI + Sarvam AI)
- **Math Agent** — joins as a LiveKit participant, listens for transcriptions, classifies and solves math queries independently using Groq (LLaMA 3.1)
- **Memory Service** — FastAPI microservice that extracts and stores user facts using Sentence Transformers + Qdrant, retrieves relevant context for each query
- **Server** — FastAPI backend managing rooms, invite links, and LiveKit token generation via Redis

## Tech Stack

- **Python** + asyncio
- **LiveKit** — real-time audio/WebRTC
- **OpenAI API** — LLM + TTS
- **Sarvam AI** — STT (Speech to Text)
- **Groq (LLaMA 3.1)** — Math Agent LLM
- **Qdrant** — vector database for memory
- **Sentence Transformers** — embeddings (all-MiniLM-L6-v2)
- **FastAPI** + uvicorn — backend services
- **Redis** — room and session management
- **SQLite** — conversation history
- **Docker Compose** — full multi-service containerization

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- API keys for OpenAI, Sarvam AI, Groq, and LiveKit

### Setup

1. Clone the repo
```bash
git clone https://github.com/vidithap/VoiceBot.git
cd VoiceBot
```

2. Copy the example env file and fill in your API keys
```bash
cp .env.example .env
```

3. Edit `.env` with your keys:
```
OPENAI_API_KEY=
SARVAM_API_KEY=
GROQ_API_KEY=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
LIVEKIT_URL=ws://localhost:7880
```

4. Start all services
```bash
docker compose up --build
```

### Connecting

**Option 1 — Browser client (recommended)**

Open `client.html` in your browser. It will connect automatically and enable your microphone.

**Option 2 — LiveKit playground**

Generate a token:
```bash
python generate_token.py
```
Then go to [meet.livekit.io](https://meet.livekit.io), paste the server URL and token, and connect.

## Project Structure

```
VoiceBot/
├── agent.py              # Main voice agent
├── math_participant.py   # Math Specialist Agent
├── memory_service.py     # Memory microservice (FastAPI + Qdrant)
├── server.py             # Room and invite management (FastAPI)
├── client.html           # Browser frontend
├── generate_token.py     # Token generator for testing
├── dockerCompose.yaml    # Multi-service Docker setup
├── Dockerfile
├── livekit.yaml          # LiveKit server config
├── egress.yaml           # Recording config
├── requirements.txt
└── .env.example
```

## Notes

- The Math Agent communicates with the Main Agent via targeted LiveKit data packets — the browser never sees these internal messages
- Memory extraction uses an LLM to decide what's worth remembering, deduplicates using cosine similarity, and updates existing memories when corrected
- Conversation recordings are saved to the `recordings/` folder via LiveKit Egress
