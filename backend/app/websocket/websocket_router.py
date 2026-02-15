from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.websocket.connection_manager import ConnectionManager
from app.multimodal.streaming_pipeline import StreamingPipeline

router = APIRouter()
manager = ConnectionManager()


@router.websocket("/ws/multimodal")
async def multimodal_websocket(websocket: WebSocket):
    await manager.connect(websocket)

    # Create NEW pipeline per connection
    pipeline = StreamingPipeline()

    try:
        while True:
            data = await websocket.receive_json()
            response = await pipeline.process_stream(data)
            await manager.send_json(websocket, response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
