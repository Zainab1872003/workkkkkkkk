# """
# Chat endpoint with organized tool management
# """
# import json
# import logging
# import os
# from typing import Optional
# from datetime import datetime

# from fastapi import APIRouter, Form, Depends, HTTPException, Request
# from fastapi.responses import StreamingResponse, JSONResponse

# from agno.agent import Agent
# from agno.models.groq import Groq

# from middleware.api_key_auth import get_api_key
# from core.config import settings
# from agno_tools.document_tools import get_tools

# router = APIRouter(tags=["Chat"])
# logger = logging.getLogger(__name__)

# # Helper functions
# def parse_tools(tools_param: Optional[str]) -> list:
#     if not tools_param or tools_param.strip() == "":
#         return []
#     try:
#         parsed = json.loads(tools_param)
#         return parsed if isinstance(parsed, list) else [parsed]
#     except:
#         return [tools_param.strip()]

# def parse_json_field(field_value: Optional[str]) -> list:
#     if not field_value:
#         return []
#     try:
#         parsed = json.loads(field_value)
#         return parsed if isinstance(parsed, list) else []
#     except:
#         return []

# @router.post("/stream/chat")
# async def process_chat(
#     request: Request,
#     user_id: str = Form(...),
#     prompt: str = Form(...),
#     tools: Optional[str] = Form(None),
#     chat_history: Optional[str] = Form(None),
#     stream: bool = Form(True),
#     api_key: str = Depends(get_api_key)
# ):
#     """Process chat with organized tools."""
    
#     try:
#         tools_list = parse_tools(tools)
#         history_list = parse_json_field(chat_history)
        
#         logger.info(f"📨 Chat - User: {user_id}, Tools: {tools_list}")
        
#         # Build context
#         context = ""
#         if history_list:
#             context = "Previous conversation:\n"
#             for msg in history_list[-5:]:
#                 if isinstance(msg, dict):
#                     context += f"{msg.get('role')}: {msg.get('content')}\n"
#             context += "\n"
        
#         final_prompt = context + f"User query: {prompt}"
        
#         # Get tools
#         tool_functions = get_tools(tools_list, user_id) if tools_list else None
        
#         # Create agent
#         agent = Agent(
#             name="Assistant",
#             model=Groq(id=settings.GROQ_MODEL, api_key=settings.GROQ_API_KEY),
#             description="You are a helpful assistant. you have to implement all the avvailable tools to answer the user query.",
#             tools=tool_functions,
#             markdown=True
#         )
        
#         # STREAMING
#         if stream:
#             async def generate_stream():
#                 try:
#                     async for chunk in agent.arun(final_prompt, stream=True):
#                         if hasattr(chunk, 'content'):
#                             yield f"data: {json.dumps({'content': chunk.content})}\n\n"
#                     yield f"data: {json.dumps({'done': True})}\n\n"
#                 except Exception as e:
#                     yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
#             return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
#         # NON-STREAMING
#         else:
#             response = agent.run(final_prompt)
#             output = response.content if hasattr(response, 'content') else str(response)
            
#             return JSONResponse({
#                 "output": output,
#                 "metadata": {"user_id": user_id, "tools": tools_list}
#             })
    
#     except Exception as e:
#         logger.error(f"❌ Error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


# routers/chat.py
"""
Chat endpoint with streaming support - Exact OpenAPI Spec Implementation
"""
import json
import logging
import os
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Form, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from agno.agent import Agent
from agno.models.groq import Groq

from core.config import settings
from core.database import get_or_create_collection
from agno_tools.document_tools import get_tools

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def parse_tools(tools_param: Optional[str]) -> list:
    """Parse tools parameter (can be JSON array or comma-separated string)"""
    if not tools_param or tools_param.strip() == "":
        return []
    try:
        parsed = json.loads(tools_param)
        return parsed if isinstance(parsed, list) else [parsed]
    except:
        return [tools_param.strip()]


def parse_json_field(field_value: Optional[str]) -> list:
    """Parse JSON string to list"""
    if not field_value:
        return []
    try:
        parsed = json.loads(field_value)
        return parsed if isinstance(parsed, list) else []
    except:
        return []


async def load_usecase(usecase_id: str) -> Optional[dict]:
    """Load usecase from MongoDB"""
    try:
        col = await get_or_create_collection("usecases")
        usecase = await col.find_one({"id": usecase_id})
        
        if usecase:
            usecase.pop("_id", None)
            logger.info(f"✅ Loaded usecase: {usecase.get('name')}")
            return usecase
        
        logger.warning(f"⚠️ Usecase '{usecase_id}' not found")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load usecase: {e}")
        return None


# ============================================================================
# /api/stream/chat - Main Chat Endpoint (Exact OpenAPI Spec)
# ============================================================================

@router.post("/stream/chat")
async def process_chat(
    request: Request,
    # Required fields
    user_id: str = Form(..., description="User ID of the person logged into the system"),
    prompt: str = Form(..., description="The user's query/prompt text"),
    
    # Optional fields with defaults from OpenAPI spec
    tools: Optional[str] = Form(
        default='["document_retriever", "calculate_bike_ijarah"]',
        description="Tools to use for processing"
    ),
    allowed_usecases_ids: Optional[str] = Form(
        default="[]",
        description="Usecase IDs that are allowed to the user"
    ),
    chat_history: Optional[str] = Form(
        default="[]",
        description="Chat history in JSON format"
    ),
    files: Optional[List[UploadFile]] = File(
        default=[],
        description="Uploaded images"
    ),
    rerank_top_k: int = Form(
        default=4,
        description="Number of reranked outputs"
    ),
    stream: bool = Form(
        default=True,
        description="Whether to stream the response"
    ),
    tool_name: str = Form(
        default="meezangpt",
        description="AI tool to use (meezangpt or opengpt)"
    )
):
    """
    Main chat endpoint that processes user queries with AI agents.
    
    **Supports:**
    - Streaming and non-streaming responses
    - File uploads
    - Usecase selection
    - Various AI models
    - Tool integration (document_retriever, calculate_bike_ijarah, image_generation)
    
    **OpenAPI Spec Compliant**
    """
    
    try:
        # Parse parameters
        tools_list = parse_tools(tools)
        allowed_usecase_ids = parse_json_field(allowed_usecases_ids)
        history_list = parse_json_field(chat_history)
        
        logger.info("="*70)
        logger.info("💬 CHAT REQUEST (OpenAPI Spec)")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Prompt: {prompt[:100]}...")
        logger.info(f"   Tools: {tools_list}")
        logger.info(f"   Allowed Usecases: {allowed_usecase_ids}")
        logger.info(f"   Stream: {stream}")
        logger.info(f"   Tool Name: {tool_name}")
        logger.info(f"   Rerank Top K: {rerank_top_k}")
        logger.info(f"   Files: {len(files) if files else 0}")
        logger.info("="*70)
        
        # Load usecase (single usecase selection)
        selected_usecase = None
        usecase_id = None
        system_prompt = "You are a helpful AI assistant."
        
        if allowed_usecase_ids and len(allowed_usecase_ids) > 0:
            usecase_id = allowed_usecase_ids[0]  # Take first usecase
            selected_usecase = await load_usecase(usecase_id)
            
            if selected_usecase:
                system_prompt = selected_usecase.get(
                    "systemPrompt",
                    "You are a helpful AI assistant. use all tools and only provide answer what the tool return no need to think of process tolls answer just return direct answers"
                )
                logger.info(f"✅ Using usecase: {selected_usecase.get('name')}")
            else:
                logger.warning(f"⚠️ Usecase '{usecase_id}' not found, using default")
        
        # Build context from chat history
        context = ""
        if history_list:
            context = "**Previous conversation:**\n"
            for msg in history_list[-5:]:  # Last 5 messages
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    context += f"{role}: {content}\n"
            context += "\n"
        
        # Handle uploaded files (images)
        files_context = ""
        if files and len(files) > 0:
            files_context = f"\n**User uploaded {len(files)} file(s).**\n"
            logger.info(f"📎 Processing {len(files)} uploaded files")
        
        # Build final prompt
        final_prompt = context + files_context + f"**Current query:** {prompt}"
        
        # Get tool functions
        tool_functions = get_tools(tools_list, user_id) if tools_list else None
        print(tool_functions)
        
        logger.info(f"🔧 Loaded {len(tool_functions) if tool_functions else 0} tool(s)")
        
        # Create AGNO agent
        agent = Agent(
            name="Multimodel Assistant",
            model=Groq(
                id=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY
            ),
            instructions=system_prompt,
            tools=tool_functions,
            markdown=True,
            # show_tool_calls=True
        )
        
# STREAMING MODE
        if stream:
            logger.info("🌊 Starting streaming response...")
            
            async def generate_stream():
                try:
                    # Stream response from agent
                    async for chunk in agent.arun(final_prompt, stream=True):
                        if hasattr(chunk, 'content') and chunk.content:
                            # Send token as SSE
                            yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                    
                    # Send completion signal with metadata (FIXED)
                    completion_data = {
                        'done': True,
                        'usecase_id': usecase_id,
                        'metadata': {
                            'user_id': user_id,
                            'tools_used': tools_list,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                    yield f"data: {json.dumps(completion_data)}\n\n"
                    
                    logger.info("✅ Streaming completed")
                    
                except Exception as e:
                    logger.error(f"❌ Streaming error: {e}", exc_info=True)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        
        # NON-STREAMING MODE
        else:
            logger.info("📝 Processing non-streaming response...")
            
            try:
                # Run agent
                response = agent.run(final_prompt)
                output = response.content if hasattr(response, 'content') else str(response)
                
                logger.info(f"✅ Response generated ({len(output)} chars)")
                
                # Return JSON response (OpenAPI spec format)
                return JSONResponse(
                    status_code=200,
                    content={
                        "output": output,
                        "metadata": {
                            "user_id": user_id,
                            "tools_used": tools_list,
                            "timestamp": datetime.now().isoformat(),
                            "stream": False,
                            "tool_name": tool_name
                        },
                        "usecase_id": usecase_id
                    }
                )
            
            except Exception as e:
                logger.error(f"❌ Agent error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Agent processing failed: {str(e)}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )
