# backend/main.py

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import asyncio
import traceback

# Add the path to your agent's directory
sys.path.append(os.path.dirname(__file__))
from agent.orchestratoragent import AgentConfig, AzureFoundryAgent

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent once at startup
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    try:
        cfg = AgentConfig.from_env()
        if not cfg.azure_endpoint or not cfg.azure_api_key:
            raise RuntimeError(
                "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in environment."
            )
        agent = AzureFoundryAgent(cfg)
        logger.info("AzureFoundryAgent initialized successfully.")
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        # Don't raise here as it would prevent the app from starting
        logger.error("Application will continue but requests will fail")

# Pydantic models for API request/response
class AskRequest(BaseModel):
    prompt: str
    file_content: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None

class AskResponse(BaseModel):
    response: str
    graph_data: Optional[Dict[str, Any]] = None
    status: str
    
@app.post("/ask", response_model=AskResponse)
async def ask_agent(request: AskRequest):
    logger.info(f"🚀 Received /ask request with prompt: '{request.prompt[:50]}...'")
    
    if agent is None:
        raise HTTPException(
            status_code=500,
            detail="Agent not initialized"
        )
    
    try:
        if not request.prompt:
            logger.warning("Empty prompt received")
            raise HTTPException(
                status_code=400,
                detail="Prompt must be provided"
            )

        # Call the orchestratoragent's run method directly
        out = await agent.run(goal=request.prompt)

        # Format the agent's structured output for the frontend
        formatted_response = {
            "response": out["result"],
            "graph_data": None,
            "status": out["status"],
        }
        
        logger.info("🚀 Request processed successfully")
        return formatted_response

    except Exception as e:
        logger.error(f"Unexpected error in /ask: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )