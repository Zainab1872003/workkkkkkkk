# gradio_app.py
"""
Gradio UI for Multimodel Agent API - WITH STREAMING SUPPORT
"""
import gradio as gr
import requests
import json
import time
import os
from typing import Generator

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "your-api-key-here")

HEADERS = {
    "x-api-key": API_KEY
}


# ============================================================================
# Streaming Chat Function
# ============================================================================

def chat_with_agent_streaming(message, history, user_id, tools, usecase_id):
    """
    Chat with streaming - yields tokens in real-time
    """
    try:
        # Prepare chat history
        chat_history = []
        if history:
            for h in history:
                chat_history.append({"role": "user", "content": h[0]})
                chat_history.append({"role": "assistant", "content": h[1]})
        
        # Prepare request
        data = {
            "user_id": user_id,
            "prompt": message,
            "tools": json.dumps(tools),
            "allowed_usecases_ids": json.dumps([usecase_id] if usecase_id else []),
            "chat_history": json.dumps(chat_history),
            "stream": "true"  # Enable streaming
        }
        
        # Make streaming request
        response = requests.post(
            f"{API_BASE_URL}/api/stream/chat",
            data=data,
            headers=HEADERS,
            stream=True,
            timeout=60
        )
        
        # Stream tokens
        full_response = ""
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                
                # Parse SSE format: "data: {...}"
                if line_text.startswith("data: "):
                    data_str = line_text[6:].strip()
                    
                    try:
                        data_obj = json.loads(data_str)
                        
                        # Check for token
                        if "token" in data_obj:
                            token = data_obj["token"]
                            full_response += token
                            
                            # Yield incrementally for streaming effect
                            yield full_response
                        
                        # Check for completion
                        elif "done" in data_obj and data_obj["done"]:
                            break
                        
                        # Check for error
                        elif "error" in data_obj:
                            error_msg = data_obj["error"]
                            yield f"❌ Error: {error_msg}"
                            break
                    
                    except json.JSONDecodeError:
                        continue
        
        # Final yield
        if full_response:
            yield full_response
        else:
            yield "No response received from the agent."
    
    except requests.exceptions.Timeout:
        yield "⏱️ Request timed out. Please try again."
    
    except Exception as e:
        yield f"❌ Error: {str(e)}"


# ============================================================================
# Other API Functions (same as before)
# ============================================================================

def upload_documents(files, user_id, ocr_type="image"):
    """Upload documents"""
    try:
        files_payload = []
        for file in files:
            files_payload.append(
                ("files", (os.path.basename(file.name), open(file.name, "rb"), "application/octet-stream"))
            )
        
        data = {"user_id": user_id, "ocr_type": ocr_type}
        
        response = requests.post(
            f"{API_BASE_URL}/api/storage/upload",
            files=files_payload,
            data=data,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            return f"✅ Upload successful!\n\n{json.dumps(response.json(), indent=2)}"
        else:
            return f"❌ Upload failed: {response.text}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def list_usecases():
    """List all usecases"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/operations/usecases/list-usecases",
            headers=HEADERS
        )
        
        if response.status_code == 200:
            data = response.json()
            usecases = data.get("response", [])
            
            if not usecases:
                return "No usecases found.", []
            
            result = f"**Found {len(usecases)} usecase(s)**\n\n"
            usecase_list = []
            
            for i, uc in enumerate(usecases, 1):
                result += f"{i}. **{uc['name']}** (ID: {uc['id']})\n"
                result += f"   {uc.get('description', 'No description')}\n\n"
                usecase_list.append(uc['id'])
            
            return result, usecase_list
        else:
            return f"❌ Error: {response.text}", []
    except Exception as e:
        return f"❌ Error: {str(e)}", []


# ============================================================================
# Gradio Interface with Streaming
# ============================================================================

def create_gradio_app():
    """Create Gradio app with real-time streaming"""
    
    with gr.Blocks(title="Multimodel Agent Platform", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🤖 Multimodel Agent Platform
            
            AI assistant with **real-time streaming**, document management, and usecase orchestration.
            """
        )
        
        with gr.Tabs():
            # ================================================================
            # TAB 1: CHAT (WITH STREAMING)
            # ================================================================
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Chat with AI Agent (Streaming Mode)")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=500,
                            show_copy_button=True,
                            bubble_full_width=False
                        )
                        
                        with gr.Row():
                            msg = gr.Textbox(
                                label="Message",
                                placeholder="Ask me anything... (Press Enter to send)",
                                lines=2,
                                scale=4
                            )
                            send_btn = gr.Button("Send 🚀", variant="primary", scale=1)
                        
                        clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Settings")
                        
                        chat_user_id = gr.Textbox(
                            label="User ID",
                            value="user123",
                            placeholder="Enter your user ID"
                        )
                        
                        tools_selection = gr.CheckboxGroup(
                            label="🔧 Tools",
                            choices=[
                                "document_retriever",
                                "calculate_bike_ijarah",
                                "image_generation"
                            ],
                            value=["document_retriever", "calculate_bike_ijarah"],
                            info="Select tools for the agent to use"
                        )
                        
                        usecase_dropdown = gr.Dropdown(
                            label="📋 Usecase",
                            choices=[],
                            value=None,
                            info="Select a usecase (optional)",
                            allow_custom_value=True
                        )
                        
                        refresh_usecases_btn = gr.Button("🔄 Refresh Usecases", size="sm")
                        
                        gr.Markdown("---")
                        gr.Markdown("**💡 Tips:**")
                        gr.Markdown("• Enable tools to search documents")
                        gr.Markdown("• Select usecase for custom prompts")
                        gr.Markdown("• Responses stream in real-time")
                
                # ============================================================
                # Chat Functionality with Streaming
                # ============================================================
                
                def respond(message, chat_history, user_id, tools, usecase):
                    """Handle chat with streaming"""
                    if not message.strip():
                        return "", chat_history
                    
                    # Add user message immediately
                    chat_history.append([message, None])
                    
                    # Stream bot response
                    for response_chunk in chat_with_agent_streaming(
                        message, chat_history[:-1], user_id, tools, usecase
                    ):
                        # Update the last message with streaming response
                        chat_history[-1][1] = response_chunk
                        yield "", chat_history
                    
                    return "", chat_history
                
                def refresh_usecase_list():
                    """Refresh usecase dropdown"""
                    _, usecase_ids = list_usecases()
                    return gr.Dropdown(choices=usecase_ids)
                
                # Connect events
                send_btn.click(
                    respond,
                    inputs=[msg, chatbot, chat_user_id, tools_selection, usecase_dropdown],
                    outputs=[msg, chatbot]
                )
                
                msg.submit(
                    respond,
                    inputs=[msg, chatbot, chat_user_id, tools_selection, usecase_dropdown],
                    outputs=[msg, chatbot]
                )
                
                clear_btn.click(lambda: [], None, chatbot, queue=False)
                
                refresh_usecases_btn.click(
                    refresh_usecase_list,
                    outputs=usecase_dropdown
                )
            
            # ================================================================
            # TAB 2: DOCUMENT MANAGEMENT
            # ================================================================
            with gr.Tab("📄 Documents"):
                gr.Markdown("### Document Management")
                
                with gr.Tab("Upload"):
                    gr.Markdown("Upload documents for AI to search through")
                    upload_user_id = gr.Textbox(label="User ID", value="user123")
                    upload_files = gr.File(
                        label="Select Files",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".jpg", ".png"]
                    )
                    ocr_type = gr.Radio(
                        ["image", "document"],
                        label="OCR Type",
                        value="image",
                        info="Choose 'image' for scanned documents"
                    )
                    upload_btn = gr.Button("📤 Upload Documents", variant="primary")
                    upload_output = gr.Markdown(label="Result")
                    
                    upload_btn.click(
                        upload_documents,
                        inputs=[upload_files, upload_user_id, ocr_type],
                        outputs=upload_output
                    )
            
            # ================================================================
            # TAB 3: STATUS
            # ================================================================
            with gr.Tab("📊 Status"):
                gr.Markdown("### API Status & Health")
                
                status_output = gr.Markdown(label="Status")
                refresh_status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                
                def get_status():
                    try:
                        response = requests.get(f"{API_BASE_URL}/health-check", headers=HEADERS)
                        if response.status_code == 200:
                            data = response.json()
                            return f"""
✅ **API Status: Online**

**Response:**
{json.dumps(data, indent=2)}

**Configuration:**
- API Base URL: `{API_BASE_URL}`
- Streaming: Enabled ✓
- Tools: Available ✓
                            """
                        else:
                            return f"❌ API Error: {response.status_code}"
                    except Exception as e:
                        return f"❌ Connection Error: {str(e)}\n\nMake sure FastAPI server is running on `{API_BASE_URL}`"
                
                refresh_status_btn.click(get_status, outputs=status_output)
                app.load(get_status, outputs=status_output)
        
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: gray;">
            <strong>Multimodel Agent Platform</strong> | Built with FastAPI + Gradio + AGNO | Powered by Groq
            </div>
            """
        )
    
    return app


# ============================================================================
# Launch
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Starting Gradio Interface for Multimodel Agent API")
    print("="*70)
    print(f"📍 API Base URL: {API_BASE_URL}")
    print(f"🔑 API Key: {'✓ Configured' if API_KEY else '✗ Missing'}")
    print("="*70 + "\n")
    
    app = create_gradio_app()
    app.queue()  # Enable queueing for streaming
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        inbrowser=True
    )
