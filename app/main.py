from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pdfplumber
import google.generativeai as genai
import json
import os
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any
import datetime

# Configure from env vars
GEMINI_KEY = os.getenv("GEMINI_KEY", "")
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "/app/serviceAccount.json")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_KEY environment variable required")

genai.configure(api_key=GEMINI_KEY)

# Initialize Firestore
def init_firestore(path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore(FIREBASE_CRED_PATH)

app = FastAPI(title="Resume Parser API")

def extract_resume_text_from_bytes(file_bytes: bytes) -> str:
    # pdfplumber expects a file-like; we write to temp file here
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = []
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text.append(t)
    return "\n".join(text).strip()

def parse_resume_with_gemini(text: str) -> Dict[str, Any]:
    prompt = f"""
Return JSON only. Follow this structure exactly:

{{
  "uid": "",
  "name": "",
  "email": "",
  "usn": "",
  "phone": "",
  "gender": "",
  "dob": "",
  "photoUrl": "",
  "academic": {{
    "collegeName": "",
    "rollno": "",
    "degree": "",
    "branch": "",
    "currentSemester": 0,
    "yearOfGraduation": 0,
    "cgpa": 0.0,
    "tenthPercentage": 0.0,
    "twelfthPercentage": 0.0,
    "backlogs": 0
  }},
  "skills": {{
    "technical": [],
    "softSkills": [],
    "tools": []
  }},
  "projects": [
    {{
      "name": "",
      "description": "",
      "techStack": [],
      "link": "",
      "duration": ""
    }}
  ],
  "experience": [
    {{
      "company": "",
      "role": "",
      "duration": "",
      "description": ""
    }}
  ],
  "certifications": [
    {{
      "title": "",
      "organization": "",
      "date": "",
      "credentialId": ""
    }}
  ],
  "preferences": {{
    "jobType": "",
    "preferredRoles": [],
    "preferredLocations": [],
    "expectedSalary": 0,
    "noticePeriod": ""
  }},
  "resume": {{
    "fileUrl": "",
    "uploadedAt": "",
    "parsedData": false
  }},
  "documents": {{
    "aadharUrl": "",
    "marksheetUrl": "",
    "offerLetters": []
  }},
  "placementStatus": {{
    "appliedJobs": [],
    "shortlistedJobs": [],
    "interviewScheduled": [
      {{"jobId": "", "date": "", "mode": ""}}
    ],
    "offers": [
      {{"company": "", "ctc": "", "offerDate": ""}}
    ]
  }},
  "meta": {{
    "status": "",
    "createdAt": "",
    "updatedAt": ""
  }}
}}

Extract only what exists in resume text. No markdown. No explanations.

Resume:
{text}
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    raw = response.text.strip()

    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])

    return json.loads(raw)

def store_student(client, data: Dict[str, Any]) -> str:
    doc_id = data.get("email") or data.get("uid")
    if not doc_id:
        raise ValueError("No email or uid found in parsed resume JSON")
    client.collection("student").document(doc_id).set(data)
    return doc_id

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    # Ensure correct content type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    body = await file.read()
    try:
        text = extract_resume_text_from_bytes(body)
        if not text:
            raise HTTPException(status_code=400, detail="PDF had no extractable text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    try:
        parsed = parse_resume_with_gemini(text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Gemini returned non-JSON or malformed JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    # add metadata
    parsed.setdefault("meta", {})
    parsed["meta"]["createdAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    parsed["resume"]["uploadedAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        doc_id = store_student(db, parsed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore error: {e}")

    return JSONResponse(status_code=201, content={"doc_id": doc_id, "message": "Stored"})

@app.get("/")
def root():
    return {"status": "ok"}
