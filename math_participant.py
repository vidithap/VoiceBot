import os
import json
import logging

class IgnoreNoisyStreamsFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "ignoring" in msg and ("byte stream" in msg or "text stream" in msg):
            return False
        return True

logging.getLogger().addFilter(IgnoreNoisyStreamsFilter())

import asyncio
import httpx
from livekit import rtc, api as lk_api

MATH_AGENT_IDENTITY = "math-specialist"


async def is_math_query(user_text: str) -> bool:
    """Lightweight LLM classifier: is this a math query?"""
    if len(user_text.split()) < 2:
        return False

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return False

    prompt = f"""Is the following message a mathematical query, calculation, equation, algebra, trigonometry, calculus, or any numerical computation request?
Reply with only YES or NO.

CRITICAL RULE: Any queries asking about EMI, loan calculations, monthly installments, or interest rates for loans are NOT math queries. They are loan questions and MUST return NO.

Message: "{user_text}"
"""

    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": 5
                },
                timeout=8.0
            )
            if res.status_code == 200:
                answer = res.json()["choices"][0]["message"]["content"].strip().upper()
                print(f"MATH AGENT: classifier decision: {answer}")
                return "YES" in answer
        except Exception as e:
            print(f"MATH AGENT: classifier error: {e}")
    return False


class MathParticipant:
    """
    Math Agent as a real LiveKit room participant (data-only, no STT/TTS).

    Works independently — not controlled by Main Agent:
    1. Listens for TRANSCRIPTION data packets sent privately to it.
    2. Independently classifies if the text is a math query.
    3. If math: solves it directly via Groq LLM and pushes a MATH_RESULT targeted data packet back to the Main Agent.
    4. If not math: sends an empty targeted MATH_RESULT so Main Agent doesn't wait forever.
    """

    def __init__(self):
        self.room = rtc.Room()

    async def connect(self, room_name: str):
        livekit_url = os.getenv("LIVEKIT_URL")
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")

        token = (
            lk_api.AccessToken(api_key, api_secret)
            .with_identity(MATH_AGENT_IDENTITY)
            .with_name("Math Specialist")
            .with_grants(lk_api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        await self.room.connect(livekit_url, token)
        print("MATH AGENT: Joined room as participant")

        # Listen independently for transcription data packets from the room
        self.room.on("data_received", self._on_data_received)
        print("MATH AGENT: Listening for transcription data packets")

    def _on_data_received(self, data):
        """
        Fires when any data packet arrives.
        Independently checks if it's a TRANSCRIPTION and handles it.
        """
        try:
            raw = data.data if hasattr(data, "data") else data
            payload = json.loads(raw.decode())
            if payload.get("type") == "TRANSCRIPTION":
                user_text = payload.get("text", "").strip()
                if user_text:
                    print(f"MATH AGENT: Received transcription, processing independently...")
                    sender_identity = data.participant.identity if (hasattr(data, "participant") and data.participant) else None
                    asyncio.create_task(self._process(user_text, sender_identity))
        except Exception as e:
            print(f"MATH AGENT: data_received error: {e}")

    async def _process(self, user_text: str, sender_identity: str = None):
        """
        Core independent logic:
        - Classify if math
        - If yes: solve directly using Groq API (no external helper function)
        - Push the result back targeted to the sender (Main Agent)
        """
        solution = ""
        try:
            if await is_math_query(user_text):
                # Call Groq LLM directly inside the agent class
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    solution = "Error: GROQ_API_KEY not found."
                else:
                    prompt = f"""You are an expert Math Specialist Agent for a voice assistant.
Your job is to solve the following math query or request.
Provide a highly concise, direct, and conversational answer.
Crucially, keep it extremely brief (maximum 1-2 sentences) so it is natural and fast for a voice assistant to read out loud.

CRITICAL FORMATTING RULE FOR TEXT-TO-SPEECH COMPATIBILITY:
- You MUST write the entire response using ONLY plain English words.
- NEVER use mathematical operator symbols (like +, -, *, /, =). Instead, write the words: "plus", "minus", "times", "divided by", "equals".
- NEVER use parentheses ( ), brackets, or raw math notation (e.g. sin(theta), sqrt(2)). Instead, write: "sine theta", "the square root of two".
- Avoid LaTeX, equations, and math formatting. Write everything as conversational sentences.
Failure to do this will crash the text-to-speech engine.

User Math Query: "{user_text}"
"""
                    async with httpx.AsyncClient() as client:
                        res = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {api_key}"},
                            json={
                                "model": "llama-3.1-8b-instant",
                                "messages": [{"role": "user", "content": prompt}],
                                "temperature": 0.0
                            },
                            timeout=15.0
                        )
                        if res.status_code == 200:
                            solution = res.json()["choices"][0]["message"]["content"].strip()
                            print(f"MATH AGENT: Solved math directly: {solution}")
                        else:
                            print(f"MATH AGENT: API error: {res.status_code}")
                            solution = "I'm having trouble solving this math problem right now."

                print(f"MATH AGENT: Pushing solution back to Main Agent ({sender_identity})")
            else:
                print(f"MATH AGENT: Not a math query, sending empty result back to Main Agent ({sender_identity})")

            # Push result back TARGETED directly to the Main Agent (so browser doesn't see it)
            result_packet = json.dumps({"type": "MATH_RESULT", "solution": solution})
            destinations = [sender_identity] if sender_identity else None
            await self.room.local_participant.publish_data(
                result_packet.encode(),
                destination_identities=destinations
            )

        except Exception as e:
            print(f"MATH AGENT: process error: {e}")
            # Send empty result targeted so Main Agent is not blocked
            try:
                fallback = json.dumps({"type": "MATH_RESULT", "solution": ""})
                destinations = [sender_identity] if sender_identity else None
                await self.room.local_participant.publish_data(
                    fallback.encode(),
                    destination_identities=destinations
                )
            except Exception:
                pass

    async def disconnect(self):
        await self.room.disconnect()
        print("MATH AGENT: Disconnected from room")
