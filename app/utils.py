import os
import tempfile
from fastapi import HTTPException
import requests
import fitz
import docx
from email import policy
from email.parser import BytesParser

def download_file(url: str) -> str:
    r = requests.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download {url}")
    suffix = os.path.splitext(url)[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(r.content)
    temp_file.close()
    return temp_file.name

from urllib.parse import urlparse

def extract_text_from_file(filepath: str) -> str:
    clean_path = filepath.split("?")[0]  
    ext = os.path.splitext(clean_path)[-1].lower()

    if ext == ".pdf":
        text = ""
        pdf_doc = fitz.open(filepath)
        for page in pdf_doc:
            text += page.get_text()
        return text
    
    elif ext == ".docx":
        doc = docx.Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    elif ext in (".eml", ".email"):
        with open(filepath, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return msg.get_body(preferencelist=('plain')).get_content()

    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Allowed: pdf, docx, email")
