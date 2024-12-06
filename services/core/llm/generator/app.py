import asyncio
import time
import random
from threading import Thread

from loguru import logger
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from contextlib import asynccontextmanager
import torch

class EmbeddedInput(BaseModel):
    """
    Creates an embedding vector representing the input text.

    Attributes:
        input (str): The text to be embedded.
        model (str): The name of the model to use for embedding.
    """
    input: str
    model: str = "mock-gpt-model"

class EmbeddedOutput(BaseModel):
    """
    Represents the embedding response

    Attributes:
        index (int): The index of the input text.
        embedding (List[float]): The embedding vector for the input text.
        object (str): The object type (default "embedding").
    """
    index: int
    embedding: List[float]
    object: str = "embedding"

class GeneratedInput(BaseModel):
    """
    Represents individual chat messages.

    Attributes:
        role (str): The role of the sender (e.g., user, assistant, system).
        content (str): The message content.
        generation_params (dict): Parameters for text generation.
    """
    role: str
    content: str
    generation_params: Dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "stream": False,
        "stop": None,
    }

class GeneratedOutput(BaseModel):
    """
    Represents the input request for chat completions.

    Attributes:
        model (str): Model name.
        messages (List[GeneratedInput]): List of chat messages.
        max_tokens (Optional[int]): Maximum number of tokens for the response.
        temperature (Optional[float]): Sampling temperature.
        stream (Optional[bool]): Whether to stream the response token by token.
    """
    model: str = "mock-gpt-model"
    messages: List[GeneratedInput]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

# ==========================
# FastAPI Application Setup
# ==========================
ml_models: Dict = {}

model_id = "meta-llama/Meta-Llama-3-8B"
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI app lifecycle.
    Initializes the ML model and tokenizer when the app starts
    and cleans up resources when the app stops.
    """
    model_name = model_id  # Replace with a valid model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token="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"
        ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token="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")
    model.eval()

    ml_models["model"] = model
    ml_models["tokenizer"] = tokenizer
    logger.info("Model and tokenizer initialized.")
    yield

    ml_models.clear()
    logger.info("Cleaned up ML resources.")

app = FastAPI(title="OpenAI-compatible API", lifespan=lifespan)

@app.post("/v1/embeddings", response_model=List[EmbeddedOutput])
async def create_embeddings(request: EmbeddedInput):
    """
    Mocks the OpenAI embeddings API.

    Args:
        request (EmbeddedInput): Input data containing the text and model name.

    Returns:
        List[EmbeddedOutput]: A mock response with random embeddings.
    """
    random.seed(len(request.input))  # Ensures deterministic mock responses
    mock_embedding = [random.uniform(-1, 1) for _ in range(768)]  # Example: 768 dimensions

    return [
        EmbeddedOutput(
            index=0,
            embedding=mock_embedding,
            object="embedding"
        )
    ]

messages: List[Dict[str, str]] = []

async def generate_response(prompt: str, stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate a text response from the modoiel, either as a complete string or streamed token-by-token.
    """
    if "model" not in ml_models or "tokenizer" not in ml_models:
        logger.error("Model not initialized.")
        raise HTTPException(status_code=500, detail="Model not initialized.")

    model = ml_models["model"]
    tokenizer = ml_models["tokenizer"]

    input_text = f"User: {prompt}\nAssistant:"
    model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    generation_kwargs = dict(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],  # Add attention mask
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

    if not stream:
        output = model.generate(**generation_kwargs)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

    streamer = TextIteratorStreamer(
        tokenizer, 
        skip_special_tokens=True,
        skip_prompt=True
    )
    generation_kwargs["streamer"] = streamer

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def token_generator():
        for token in streamer:
            yield token

    return token_generator()

@app.get("/")
async def root():
    """Root endpoint to check if the service is running."""
    return {"message": "Hello, World"}

@app.post("/chat/completions")
async def generate(request: GeneratedOutput):
    prompt = request.messages[-1].content

    if request.stream:
        async def token_generator():
            async for token in await generate_response(prompt, stream=True):
                yield token

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream"
        )

    generated_text = await generate_response(prompt)  # Ensure this line uses await
    return {
        "id": "completion-id",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"message": {"role": "assistant", "content": generated_text}}],
    }

# Testing if model is runnign with GPU or CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)