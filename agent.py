import os
import logging

class IgnoreNoisyStreamsFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "ignoring" in msg and ("byte stream" in msg or "text stream" in msg):
            return False
        return True

logging.getLogger().addFilter(IgnoreNoisyStreamsFilter())

import time
import asyncio
import json
import httpx
from datetime import datetime
from dotenv import load_dotenv
MEMORY_SERVICE_URL = os.getenv("MEMORY_SERVICE_URL", "http://localhost:8001")
print(f"DEBUG: MEMORY_SERVICE_URL is set to {MEMORY_SERVICE_URL}")

async def save_message(user_id, room_id, sender, message):
    async with httpx.AsyncClient() as client:
        try:
            start = time.time()
            await client.post(f"{MEMORY_SERVICE_URL}/messages", json={
                "user_id": user_id,
                "room_id": room_id,
                "sender": sender,
                "message": message
            })
            print(f"save_message latency: {time.time() - start:.3f}s")
        except Exception as e:
            print(f"save_message error: {e}")

async def get_recent_messages(user_id, room_id, limit=6):
    async with httpx.AsyncClient() as client:
        try:
            start = time.time()
            res = await client.get(f"{MEMORY_SERVICE_URL}/messages", params={
                "user_id": user_id, "room_id": room_id, "limit": limit
            })
            print(f"get_recent_messages latency: {time.time() - start:.3f}s")
            data = res.json()
            return [(m[0], m[1]) for m in data.get("messages", [])]
        except Exception as e:
            print(f"get_recent_messages error: {e}")
            return []

async def retrieve_context(user_id, query):
    async with httpx.AsyncClient() as client:
        try:
            start = time.time()
            res = await client.post(f"{MEMORY_SERVICE_URL}/memory/search", json={
                "user_id": user_id, "query": query
            })
            print(f"retrieve_context latency: {time.time() - start:.3f}s")
            data = res.json()
            return data.get("context", "")
        except Exception as e:
            print(f"retrieve_context error: {e}")
            return ""

def write_log(file_name, message):
    try:
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception as e:
        print("FILE LOG ERROR:", e)

def log_msg(event, message):
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    return f"[{current_time}] {event}: {message}"

from livekit import agents, api
from livekit.agents import AgentServer, AgentSession, Agent, room_io, function_tool
from livekit.plugins import silero, groq
from duckduckgo_search import DDGS
from math_participant import MathParticipant

load_dotenv()

egress_id = None

async def extract_and_store_memory(user_id: str, user_text: str):
    async with httpx.AsyncClient() as client:
        try:
            start = time.time()
            res = await client.post(f"{MEMORY_SERVICE_URL}/memory/extract", json={
                "user_id": user_id, "user_text": user_text
            }, timeout=10.0)
            print(f"extract_and_store_memory latency: {time.time() - start:.3f}s")
        except Exception as e:
            print(f"extract_and_store_memory error: {e}")

def inject_assistant_note(chat_ctx, content):
    """
    Safely injects an assistant note into the chat context, positioning it
    directly BEFORE the user's latest query so it guides the LLM without being spoken.
    """
    chat_ctx.add_message(role="assistant", content=content)
    msgs = chat_ctx.messages() if callable(chat_ctx.messages) else chat_ctx.messages
    user_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if getattr(msgs[i], "role", None) == "user":
            user_idx = i
            break
    if user_idx != -1:
        note_msg = msgs.pop()
        msgs.insert(user_idx, note_msg)

async def observer_check(user_id: str, user_text: str):
    try:
        if len(user_text.split()) < 3:
            return None
            
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: return None
        
        start = time.time()
        async with httpx.AsyncClient() as client:
            # 1. Fetch user's past messages across all sessions
            try:
                history_res = await client.get(f"{MEMORY_SERVICE_URL}/messages", params={
                    "user_id": user_id, "limit": 20
                })
                history_data = history_res.json().get("messages", [])
                history_str = "\n".join([f"{sender}: {msg}" for sender, msg in history_data])
            except Exception as he:
                print(f"History retrieval error: {he}")
                history_data = []
                history_str = ""

            has_bot_correction = any(sender == "bot" for sender, msg in history_data)

            if not has_bot_correction:
                print("Observer: no bot corrections in history -> SEARCH-only mode")
                search_prompt = f"""The user said: "{user_text}"

Is this a verifiable factual claim (e.g. geography, science, history, finance)?
--- If YES: reply exactly: SEARCH: <2-3 word search query>
--- If NO (conversational, greeting, opinion, mathematical calculation, trigonometry, algebra, calculus, or equation request): reply exactly: NONE

CRITICAL RULE: Any math queries, questions about math equations, sin, cos, theta, integrals, derivatives, or algebra problems are NOT factual claims. They are calculation requests and MUST return NONE.

Reply with ONLY the result."""

                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": search_prompt}],
                        "temperature": 0.0
                    },
                    timeout=10.0
                )
                if res.status_code != 200:
                    print(f"Observer SEARCH-only error: {res.status_code}")
                    return None

                decision = res.json()["choices"][0]["message"]["content"].strip()
                print(f"Observer SEARCH-only decision: {decision}")

                if "NONE" in decision or "SEARCH:" not in decision:
                    return None

                query = decision.replace("SEARCH:", "").strip()

            else:
                # Bot corrections exist in history → full RECALL vs SEARCH classification
                if not history_str.strip():
                    return None

                full_prompt = f"""Review this past conversation history:
{history_str}

The user just said: "{user_text}"

Classify into one of:
1. RECALL: The user is actively and explicitly asserting the SAME WRONG FACTUAL CLAIM that the bot already corrected in history.
   - The user must be making a direct factual statement/assertion, not just chatting, asking a question, or referencing it.
   - Example of RECALL: "No, Sydney is definitely the capital of Australia."
   - OUTPUT: RECALL: <brief recall of what was corrected>
2. SEARCH: A direct factual statement/assertion of fact made by the user that is not yet corrected. Do NOT classify general questions, math problems, equations, or calculations as SEARCH.
   - OUTPUT: SEARCH: <2-3 word search query>
3. NONE: Anything conversational — questions about a topic, greetings, farewells, acknowledgements, thanks, opinions, filler, mathematical calculations, trigonometry, algebra, calculus, or equation requests.
   - Examples of NONE: "Okay", "Thank you", "Bye", "Got it", "I see", "No, thank you", "Sure", "What's the impact of that?", "Why?", "What do you mean?", "We'll get some..."
   - OUTPUT: NONE

CRITICAL RULES:
- Any math queries, questions about math equations, sin, cos, theta, integrals, derivatives, or algebra problems are NOT factual claims. They are calculation requests and MUST return NONE.
- If the user's message is a question (e.g., starts with "What", "Why", "How", or asks a query), you MUST classify as NONE or SEARCH, never RECALL.
- RECALL only applies when the user is actively repeating their incorrect assertion.
- Acknowledgements, thanks, and standard conversation are ALWAYS NONE.
- When in doubt, choose NONE over RECALL.
Reply with ONLY the result."""

                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": full_prompt}],
                        "temperature": 0.0
                    },
                    timeout=10.0
                )
                if res.status_code != 200:
                    print(f"Observer Error: {res.status_code} - {res.text}")
                    return None

                decision = res.json()["choices"][0]["message"]["content"].strip()
                print(f"Observer full decision: {decision}")

                if "NONE" in decision or not decision:
                    return None

                if "RECALL:" in decision:
                    nudge = decision.replace("RECALL:", "").strip()
                    print(f"observer_check (recall) latency: {time.time() - start:.3f}s")
                    return f"PAST_RECALL: {nudge}"

                if "SEARCH:" not in decision:
                    return None

                query = decision.replace("SEARCH:", "").strip()

            # --- Web Search (shared by both paths) ---
            try:
                results = DDGS().text(query, max_results=2)
            except Exception as se:
                print(f"Search error: {se}")
                return None

            check_prompt = f"User said: '{user_text}'. Internet facts: {str(results)}. Is the user factually incorrect? If they are correct or it's subjective, reply exactly 'OK'. If they are wrong, write a brief correction (1-2 sentences)."
            res2 = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": check_prompt}],
                    "temperature": 0.0
                },
                timeout=10.0
            )
            if res2.status_code != 200:
                print(f"Observer Step 3 Error: {res2.status_code} - {res2.text}")
                return None

            correction = res2.json()["choices"][0]["message"]["content"].strip()
            print(f"observer_check (web search) latency: {time.time() - start:.3f}s")
            if correction == "OK" or "OK" in correction:
                return None
            return correction

    except Exception as e:
        print("observer error:", e)
        return None

# ------------------ SERVICES ------------------

class Services:
    def __init__(self):
        self.stt = groq.STT(
            model="whisper-large-v3-turbo",
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.llm = groq.LLM(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
        )

        self.tts = groq.TTS(
            api_key=os.getenv("GROQ_API_KEY"),
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
        )

        self.vad = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.2
        )

# ------------------ EGRESS ------------------

async def start_recording(room_name: str):
    try:
        lk_api = api.LiveKitAPI(
            url=os.getenv("LIVEKIT_URL", "http://localhost:7880"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )

        request = api.RoomCompositeEgressRequest(
            room_name=room_name,
            audio_only=True,
            audio_mixing=api.AudioMixing.DUAL_CHANNEL_AGENT,
            file_outputs=[
                api.EncodedFileOutput(
                    filepath=f"/out/{room_name}.ogg",
                    file_type='OGG'
                )
            ],
        )
        print("REQUEST TYPE:", type(request))
        print("REQUEST DATA:")
        print(request)

        res = await lk_api.egress.start_room_composite_egress(request)
        global egress_id
        egress_id = res.egress_id
        print("Recording started:", res.egress_id)
        await lk_api.aclose()

    except Exception as e:
        print("Egress error:", e)

# ------------------ AGENT ------------------

class Assistant(Agent):
    def __init__(self, file_name, room_name, user_id) -> None:
        super().__init__(
            instructions="""
            You are a helpful loan assistant.

            You operate alongside a Math Specialist Agent and a background fact-checking component:

            1. FACT-CHECKER (background): You will occasionally receive silent corrections injected into your chat history as assistant messages starting with "OBSERVER NOTE: ...". NEVER speak the words "OBSERVER NOTE", never repeat or reveal these to the user. Silently adopt the corrected facts as your own knowledge and naturally correct the user in your response. When correcting, do NOT say "You're right" — politely correct them directly.

            2. MATH SPECIALIST AGENT: There is a dedicated Math Specialist Agent running alongside you that independently handles ALL mathematical queries (calculations, equations, algebra, trigonometry, calculus, any numerical computation).
               - IMPORTANT: Do NOT attempt to solve math queries yourself. The Math Specialist Agent will always handle these and inject the result.
               - You will receive the solution as an assistant message starting with "MATH NOTE: ...". Present it naturally and conversationally as if you worked it out yourself.
               - NEVER speak the words "Math Specialist", "Math Agent", or "MATH NOTE" to the user.

            EMI Calculation:
            - If the user asks about EMI or loan calculation, use the `calculate_emi` tool.
            - Use memory if values are available, otherwise ask for missing values first.

            For everything else, respond naturally as a helpful loan assistant.
            """
        )
        self.last_agent_text = None
        self.file_name = file_name
        self.room_name = room_name
        self.user_id = user_id
        self.room = None
        # Event + storage for math result pushed by Math Agent independently
        self._math_result_event = asyncio.Event()
        self._math_result = None

    @function_tool
    async def calculate_emi(self, principal: float, rate: float, time: float) -> str:
        """Calculate the monthly EMI (Equated Monthly Installment) for a loan.

        Args:
            principal: The loan amount (in rupees).
            rate: The annual interest rate as a percentage (e.g., 6.5 for 6.5%).
            time: The loan duration in years.
        """
        if principal <= 0:
            return "Please provide a valid loan amount."

        if rate <= 0:
            return "Please provide a valid interest rate."

        if time <= 0:
            return "Please provide a valid loan duration."

        msg = log_msg("TOOL CALLED", f"calculate_emi(principal={principal}, rate={rate}, time={time})")
        print(msg)
        write_log(self.file_name, msg)

        monthly_rate = rate / (12 * 100)
        months = time * 12

        emi = (principal * monthly_rate * (1 + monthly_rate) ** months) / \
              ((1 + monthly_rate) ** months - 1)

        result = f"Your monthly EMI will be {round(emi, 2)} rupees."

        msg = log_msg("TOOL RESULT", result)
        print(msg)
        write_log(self.file_name, msg)

        return result



    def set_math_result(self, solution: str):
        """Called by the room data listener when Math Agent pushes a result."""
        self._math_result = solution
        self._math_result_event.set()

    async def on_user_turn_completed(self, chat_ctx, new_message=None):
        try:

            await asyncio.sleep(0.6)
            total_start = time.time()
            if new_message and hasattr(new_message, "content"):
                user_text = (
                    " ".join(map(str, new_message.content))
                    if isinstance(new_message.content, list)
                    else str(new_message.content)
                ).strip()

                msg = log_msg("USER", user_text)
                print(msg)
                write_log(self.file_name, msg)
                
                asyncio.create_task(save_message(self.user_id, self.room_name, "user", user_text))

                # --- Recent conversation memory ---
                recent_msgs = await get_recent_messages(self.user_id, self.room_name)

                print("RECENT MSGS:", recent_msgs)

                if recent_msgs:
                    context = "\n".join([f"{s}: {m}" for s, m in recent_msgs])

                    chat_ctx.add_message(
                        role="system",
                        content=f"Recent conversation:\n{context}"
                    )

                # --- Reset math result state for this turn ---
                self._math_result_event.clear()
                self._math_result = None

                # --- Send transcription to Math Specialist ---
                if self.room:
                    packet = json.dumps({"type": "TRANSCRIPTION", "text": user_text})
                    await self.room.local_participant.publish_data(
                        packet.encode(),
                        destination_identities=["math-specialist"]
                    )

                # --- Memory and observer run in parallel while Math Agent processes independently ---
                memory_context, observer_correction = await asyncio.gather(
                    retrieve_context(self.user_id, user_text),
                    observer_check(self.user_id, user_text),
                )

                # --- Wait for Math Agent to push its result (it works independently) ---
                try:
                    await asyncio.wait_for(self._math_result_event.wait(), timeout=10.0)
                    math_solution = self._math_result or ""
                except asyncio.TimeoutError:
                    math_solution = ""
                    print("ORCHESTRATOR: Math Agent timeout, proceeding without math result")

                print(f"ORCHESTRATOR: math_detected={bool(math_solution)}")

                if memory_context:
                    print("MEMORY CONTEXT:", memory_context)
                    chat_ctx.add_message(
                        role="system",
                        content=f"Relevant memories:\n{memory_context}"
                    )

                if math_solution:
                    print(f"MATH SPECIALIST SOLUTION: {math_solution}")
                    inject_assistant_note(
                        chat_ctx,
                        f"MATH NOTE: The Math Specialist has already solved this. Solution: {math_solution}. Present this to the user naturally and conversationally."
                    )
                elif observer_correction:
                    print("OBSERVER NUDGE:", observer_correction)
                    if observer_correction.startswith("PAST_RECALL:"):
                        verified_fact = memory_context.strip() if memory_context else "The capital of Australia is Canberra, not Sydney."
                        inject_assistant_note(
                            chat_ctx,
                            f"OBSERVER NOTE: The user is repeating a factual mistake that was already corrected in a previous session. Verified fact: {verified_fact}. You MUST explicitly acknowledge that you both discussed and corrected this same topic in a previous session (e.g. \"As we discussed in our previous conversation...\")."
                        )
                    else:
                        inject_assistant_note(
                            chat_ctx,
                            f"OBSERVER NOTE: {observer_correction}"
                        )
                        asyncio.create_task(extract_and_store_memory(self.user_id, f"Actually, {observer_correction}"))

                asyncio.create_task(extract_and_store_memory(self.user_id, user_text))

            msgs = chat_ctx.messages() if callable(chat_ctx.messages) else chat_ctx.messages

            valid_msgs = []
            for m in msgs:
                if getattr(m, "role", None) == "assistant" and m.content:
                    content_str = (
                        " ".join(map(str, m.content))
                        if isinstance(m.content, list)
                        else str(m.content)
                    ).strip()
                    if not content_str.startswith("OBSERVER NOTE:") and not content_str.startswith("MATH NOTE:"):
                        valid_msgs.append(m)

            if valid_msgs:
                latest = valid_msgs[-1]

                agent_text = (
                    " ".join(map(str, latest.content))
                    if isinstance(latest.content, list)
                    else str(latest.content)
                ).strip()

                if agent_text and agent_text != self.last_agent_text:
                    self.last_agent_text = agent_text

                    msg = log_msg("AGENT", agent_text)
                    print(msg)
                    write_log(self.file_name, msg)
                    asyncio.create_task(save_message(self.user_id, self.room_name, "bot", agent_text))

                total_end = time.time()
                print("TOTAL TURN LATENCY:", total_end - total_start)
     
        except Exception as e:
            print("LOG ERROR:", e)

# ------------------ SERVER ------------------

server = AgentServer()

@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    file_name = f"/app/transcripts/{ctx.room.name}_transcript.txt"

    msg = log_msg("USER JOINED", participant.identity)
    print(msg)

    services = Services()

    session = AgentSession(
        stt=services.stt,
        llm=services.llm,
        tts=services.tts,
        vad=services.vad
    )

    msg = log_msg("SESSION", "STARTING")
    print(msg)
    write_log(file_name, msg)

    # --- Set up Math Agent as a room participant ---
    math_participant = MathParticipant()
    asyncio.create_task(math_participant.connect(ctx.room.name))

    assistant = Assistant(file_name, ctx.room.name, participant.identity)
    assistant.room = ctx.room

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(),
        ),
    )

    # --- Listen for data packets pushed by Math Agent ---
    @ctx.room.on("data_received")
    def on_data_received(data):
        try:
            raw = data.data if hasattr(data, "data") else data
            payload = json.loads(raw.decode())
            if payload.get("type") == "MATH_RESULT":
                solution = payload.get("solution", "")
                print(f"ORCHESTRATOR: Math Agent pushed result: {'(solution)' if solution else '(empty)'}")
                assistant.set_math_result(solution)
        except Exception as e:
            print(f"ORCHESTRATOR: data_received error: {e}")

    await start_recording(ctx.room.name)

    msg = log_msg("SESSION", "STARTED")
    print(msg)
    write_log(file_name, msg)

    @ctx.room.on("participant_disconnected")
    def on_leave(participant):

        msg1 = log_msg("USER LEFT", participant.identity)
        msg2 = log_msg("SESSION", "ENDED")

        print(msg1)
        print(msg2)

        write_log(file_name, msg2)

        async def stop_and_close():
            global egress_id

            if egress_id:
                try:
                    lk_api = api.LiveKitAPI(
                        url=os.getenv("LIVEKIT_URL"),
                        api_key=os.getenv("LIVEKIT_API_KEY"),
                        api_secret=os.getenv("LIVEKIT_API_SECRET"),
                    )

                    await lk_api.egress.stop_egress(
                        api.StopEgressRequest(egress_id=egress_id)
                    )

                    print("Recording stopped")

                except Exception as e:
                    print("Stop skipped:", e)

            await session.aclose()

        asyncio.create_task(stop_and_close())

# ------------------ ENTRY ------------------

def main():
    print("Starting VoiceBot...")
    agents.cli.run_app(server)

if __name__ == "__main__":
    main()
