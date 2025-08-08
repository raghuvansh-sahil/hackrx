import os
import tempfile
from fastapi import HTTPException
import requests
import docx
import PyPDF2
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

def extract_text_from_file(filepath: str) -> str:
    clean_path = filepath.split("?")[0]
    ext = os.path.splitext(clean_path)[-1].lower()

    if ext == ".pdf":
        text = ""
        with open(filepath, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
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