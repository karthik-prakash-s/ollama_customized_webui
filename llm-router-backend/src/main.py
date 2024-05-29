from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.llmrouter import prompt_router 
import logging
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, change this to specific origins as needed
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Define a Pydantic model for the request body
class TextRequest(BaseModel):
    text: str

logging.basicConfig(level=logging.INFO)
# Define a route to receive large text
@app.post("/get-llm")
async def process_text(request: TextRequest):
    try:
        logging.info(f"Received text: {request.text}")
        
        # Simulate a call to prompt_router
        res = prompt_router(request.text)  # Assuming this is your function call

        # Return the response
        return res

    except ValueError as ve:

        logging.error(f"ValueError occurred: {ve}")
        raise HTTPException(status_code=400, detail="Invalid input")

    except Exception as e:
        # Log the error
        logging.error(f"An error occurred: {e}")
        # Raise a 500 Internal Server Error
        raise HTTPException(status_code=500, detail="Internal Server Error")
