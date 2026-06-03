from livekit import api
from dotenv import load_dotenv  
import os
load_dotenv()

token = api.AccessToken()

token = token.with_grants( api.VideoGrants(
    room_join=True,
    room="test_room",
    can_publish=True,
    can_subscribe=True,
))

token =token.with_identity("browser-user")

jwt_token = token.to_jwt()

print("=" * 60)
print("🎤 LiveKit Token for Browser Client")
print("=" * 60)
print()
print("1. Go to: https://meet.livekit.io/")
print()
print("2. Fill in these values:")
print(f"   Server URL: ws://localhost:7880")
print(f"   Token: {jwt_token}")
print()
print("3. Click Connect → Click Microphone → Speak!")
print()
print("=" * 60)
