import os
import time
import asyncio
import httpx
import sqlite3
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

app = FastAPI(title="Memory and DB Service")

# --- Database Setup ---
os.makedirs("/app/transcripts", exist_ok=True)
db_path = os.getenv("DB_PATH", "/app/transcripts/chat.db") 
conn = sqlite3.connect(db_path, check_same_thread=False)

with conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        room_id TEXT,
        sender TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

# --- Qdrant Setup ---
embedding_model = None

qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
qdrant_client = QdrantClient(host=qdrant_host, port=6333)

if not qdrant_client.collection_exists("memories"):
    qdrant_client.create_collection(
        collection_name="memories",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

class MemU:
    def search(self, user_id, query, limit=5):
        global embedding_model
        if embedding_model is None:
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            
        query_embedding = embedding_model.encode(query)

        results = qdrant_client.query_points(
            collection_name="memories",
            query=query_embedding.tolist(),
            limit=limit
        )

        memories = []
        for result in results.points:
            payload = result.payload
            memory = payload.get("memory", "")
            if (
                payload.get("user_id") == user_id
                and len(memory.split()) >= 3
                and result.score >= 0.1   
            ):
                memories.append(memory)
        return memories
    
    def retrieve_context(self, user_id, query):
        memories = self.search(user_id, query)
        if not memories:
            return ""
        return "\n".join(memories)

memu = MemU()

# --- Pydantic Models ---
class MessageRequest(BaseModel):
    user_id: str
    room_id: str
    sender: str
    message: str

class MemoryExtractRequest(BaseModel):
    user_id: str
    user_text: str

class MemorySearchRequest(BaseModel):
    user_id: str
    query: str

# --- API Endpoints ---

@app.post("/messages")
def save_message(req: MessageRequest):
    with conn:
        conn.execute(
            "INSERT INTO conversations (user_id, room_id, sender, message) VALUES (?, ?, ?, ?)",
            (req.user_id, req.room_id, req.sender, req.message)
        )
    return {"status": "success"}

@app.get("/messages")
def get_recent_messages(user_id: str, room_id: str = None, limit: int = 6):
    cursor = conn.cursor()
    if room_id:
        cursor.execute("""
            SELECT sender, message FROM conversations
            WHERE user_id = ? AND room_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, room_id, limit))
    else:
        cursor.execute("""
            SELECT sender, message FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit))

    rows = cursor.fetchall()
    cursor.close()

    recent = rows[::-1]
    return {"messages": recent}

@app.post("/memory/extract")
async def extract_and_store_memory(req: MemoryExtractRequest):
    global embedding_model

    if embedding_model is None:
        loop = asyncio.get_event_loop()
        embedding_model = await loop.run_in_executor(
            None, SentenceTransformer, "all-MiniLM-L6-v2"
        )   

    is_correction = req.user_text.strip().startswith("Actually,")

    if is_correction:
        prompt = f"""You are a memory extractor for a loan assistant voice bot.
From the text, extract the general factual correction worth remembering in future sessions.
Return one clean sentence starting with "Fact".

Example:
"Actually, the sky is blue, not green." → Fact: The sky is blue, not green.

Text message: "{req.user_text}"
"""
    else:
        prompt = f"""You are a memory extractor for a loan assistant voice bot.
From the text, extract personal facts about the user worth remembering in future sessions (e.g., location, name, preferences).
Return one clean sentence starting with "User".
If no personal fact, reply with exactly: NONE

Examples:
"I stay in Bangalore" → User lives in Bangalore.
"Calculate EMI for 10 lakhs at 5% for 2 years" → User wants a loan of 10 lakhs at 5% interest for 2 years.
"The sun rises in the west." → NONE
"Hello" → NONE

Text message: "{req.user_text}"
"""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0,
                }
            )
            data = res.json()
            extracted = data["choices"][0]["message"]["content"].strip()
            print(f"MEMORY EXTRACTOR → '{extracted}'")

            if extracted.upper() == "NONE" or not extracted:
                print("MEMORY SKIPPED:", req.user_text[:80])
                return {"status": "skipped"}

            embedding = await asyncio.get_event_loop().run_in_executor(
                None, embedding_model.encode, extracted
            )

            search_result = qdrant_client.query_points(
                collection_name="memories",
                query=embedding.tolist(),
                limit=1
            )

            duplicate = (
                len(search_result.points) > 0
                and search_result.points[0].score >= 0.9
                and search_result.points[0].payload.get("user_id") == req.user_id
            )

            if duplicate:
                existing_id = search_result.points[0].id
                qdrant_client.upsert(
                    collection_name="memories",
                    points=[{
                        "id": existing_id,
                        "vector": embedding.tolist(),
                        "payload": {
                            "user_id": req.user_id,
                            "memory": extracted
                        }
                    }]
                )
                print("MEMORY UPDATED ✓:", extracted)
                return {"status": "updated", "memory": extracted}
            else:
                qdrant_client.upsert(
                    collection_name="memories",
                    points=[{
                        "id": int(time.time() * 1000),
                        "vector": embedding.tolist(),
                        "payload": {
                            "user_id": req.user_id,
                            "memory": extracted
                        }
                    }]
                )
                print("MEMORY STORED ✓:", extracted)
                return {"status": "stored", "memory": extracted}

    except Exception as e:
        import traceback
        print("MEMORY EXTRACTOR ERROR:", e)
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/search")
def search_memory(req: MemorySearchRequest):
    context = memu.retrieve_context(req.user_id, req.query)
    return {"context": context}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
