"""
Chat endpoint with organized tool management
"""
import json
import logging
import os
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Form, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from agno.agent import Agent
from agno.models.groq import Groq

from middleware.api_key_auth import get_api_key
from core.config import settings
from agno_tools.document_tools import get_tools

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)

# Helper functions
def parse_tools(tools_param: Optional[str]) -> list:
    if not tools_param or tools_param.strip() == "":
        return []
    try:
        parsed = json.loads(tools_param)
        return parsed if isinstance(parsed, list) else [parsed]
    except:
        return [tools_param.strip()]

def parse_json_field(field_value: Optional[str]) -> list:
    if not field_value:
        return []
    try:
        parsed = json.loads(field_value)
        return parsed if isinstance(parsed, list) else []
    except:
        return []

@router.post("/stream/chat")
async def process_chat(
    request: Request,
    user_id: str = Form(...),
    prompt: str = Form(...),
    tools: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),
    stream: bool = Form(True),
    api_key: str = Depends(get_api_key)
):
    """Process chat with organized tools."""
    
    try:
        tools_list = parse_tools(tools)
        history_list = parse_json_field(chat_history)
        
        logger.info(f"📨 Chat - User: {user_id}, Tools: {tools_list}")
        
        # Build context
        context = ""
        if history_list:
            context = "Previous conversation:\n"
            for msg in history_list[-5:]:
                if isinstance(msg, dict):
                    context += f"{msg.get('role')}: {msg.get('content')}\n"
            context += "\n"
        
        final_prompt = context + f"User query: {prompt}"
        
        # Get tools
        tool_functions = get_tools(tools_list, user_id) if tools_list else None
        
        # Create agent
        agent = Agent(
            name="Document Assistant",
            model=Groq(id=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY),
            description="You are a helpful document assistant. Use tools to search and manage documents.",
            tools=tool_functions,
            markdown=True
        )
        
        # STREAMING
        if stream:
            async def generate_stream():
                try:
                    async for chunk in agent.arun(final_prompt, stream=True):
                        if hasattr(chunk, 'content'):
                            yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                    yield f"data: {json.dumps({'done': True})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # NON-STREAMING
        else:
            response = agent.run(final_prompt)
            output = response.content if hasattr(response, 'content') else str(response)
            
            return JSONResponse({
                "output": output,
                "metadata": {"user_id": user_id, "tools": tools_list}
            })
    
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
