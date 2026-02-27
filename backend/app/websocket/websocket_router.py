from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket.connection_manager import ConnectionManager
from app.multimodal.streaming_pipeline import StreamingPipeline
import json

router = APIRouter()
#manager = ConnectionManager()


@router.websocket("/ws/multimodal")
async def multimodal_websocket(websocket: WebSocket):

    await websocket.accept()
    pipeline = StreamingPipeline()

    try:
        while True:

           message = await websocket.receive_text()

           data = json.loads(message)

           response = await pipeline.process_stream(data)

           if response:
                await websocket.send_json(response)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()
                    
    '''    message = await websocket.receive()
            
            # AUDIO (binary)
            if "bytes" in message and message["bytes"] is not None:
                response = await pipeline.process_audio_bytes(message["bytes"])
                if response:
                    await websocket.send_json(response)

            # TEXT (optional)
            elif "text" in message and message["text"] is not None:
                data = json.loads(message["text"])
                response = await pipeline.process_stream(data)
                await websocket.send_json(response)
'''
        
