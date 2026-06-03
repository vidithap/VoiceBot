from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
import os
from dotenv import load_dotenv
import redis
import uuid
from fastapi import Request

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  JOIN USING INVITE 
@app.get("/join-by-invite/{invite_id}")
def join_by_invite(invite_id: str, request: Request):

    # Get room from Redis
    room_name = r.get(f"invite:{invite_id}")

    if not room_name:
        return {"error": "Invalid invite"}

    room_key = f"room:{room_name}:users"

    # Check how many users are in room
    user_count = r.scard(room_key)

    if user_count >= 2:
        return {"error": "Room is full"}

    # Create user
    user_id = request.query_params.get("user_id")
    if not user_id:
        return {"error": "user_id missing"}
    
    r.sadd(room_key, user_id)

    # Generate LiveKit token
    token = api.AccessToken(
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET"),
    )

    token = token.with_identity(user_id)
    token = token.with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        )
    )

    return {
        "token": token.to_jwt(),
        "user_id": user_id,
        "room_name": room_name 
    }


@app.post("/leave/{room_name}/{user_id}")
def leave(room_name: str, user_id: str):
    room_key = f"room:{room_name}:users"
    r.srem(room_key, user_id)
    return {"message": "User removed"}

@app.get("/create-invite")
def create_invite():
    invite_id = str(uuid.uuid4())
    room_name = f"room_{invite_id}"

    r.set(f"invite:{invite_id}", room_name, ex=3600)

    return {"invite_id": invite_id}