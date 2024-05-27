from fastapi import FastAPI

# Create FastAPI app instance
app = FastAPI()

# Define a route to receive large text
@app.post("/process_text/")
async def process_text(text: str):
    # Process the received text here
    # For demonstration, let's just return the length of the text
    return {"text_length": len(text)}
