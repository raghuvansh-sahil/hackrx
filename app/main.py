import os
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List, Union
from utils import download_file, extract_text_from_file
from ai import AI
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
gemini_model = None

class HackRXRequest(BaseModel):
    documents: Union[HttpUrl, List[HttpUrl]]
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
def on_startup():
    global gemini_model
    genai.configure(api_key=os.getenv('API_KEY'))
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')

@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(payload: HackRXRequest):
    doc_urls = payload.documents if isinstance(payload.documents, list) else [payload.documents]

    full_text = ""
    for url in doc_urls:
        filepath = download_file(str(url))
        full_text += "\n\n" + extract_text_from_file(filepath)
        os.remove(filepath)

    ai = AI(gemini_model)
    answers = ai.process_and_answer(full_text, payload.questions)
    return {"answers": answers}