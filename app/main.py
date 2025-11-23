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
import numpy as np

# ---------- CONFIG ----------
GEMINI_KEY = os.getenv("GEMINI_KEY", "")
FIREBASE_CRED_PATH = os.getenv("FIREBASE_CRED_PATH", "/app/serviceAccount.json")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")          # your gmail ID
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")    # Gmail App Password


if not GEMINI_KEY:
    raise RuntimeError("GEMINI_KEY environment variable required")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable required")

genai.configure(api_key=GEMINI_KEY)

# ---------- FIREBASE INIT ----------
def init_firestore(path):
    if not firebase_admin._apps:
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore(FIREBASE_CRED_PATH)

# ---------- PINECONE INIT ----------
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "students-index"

# Create pinecone index if needed
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize(vec):
    vec = np.array(vec)
    return (vec / np.linalg.norm(vec)).tolist()

# ---------- FASTAPI ----------
app = FastAPI(title="Resume Parser + Pinecone API")

# ---------- PDF TEXT EXTRACTOR ----------
def extract_resume_text_from_bytes(file_bytes: bytes) -> str:
    import io
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = []
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text.append(t)
    return "\n".join(text).strip()

# ---------- GEMINI PARSER ----------
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

# ---------- FIRESTORE SAVE ----------
def store_student(client, data: Dict[str, Any]) -> str:
    doc_id = data.get("email") or data.get("uid")
    if not doc_id:
        raise ValueError("No email or uid found in parsed resume JSON")
    client.collection("student").document(doc_id).set(data)
    return doc_id

# ---------- NEW: BUILD EMBEDDING PAYLOAD ----------
def build_embedding_payload(parsed: Dict[str, Any]) -> str:
    """
    We will embed only the KEY attributes:
    - technical skills
    - CGPA
    - branch
    - degree
    - graduation year
    """
    tech = parsed.get("skills", {}).get("technical", [])
    cgpa = parsed.get("academic", {}).get("cgpa", "")
    branch = parsed.get("academic", {}).get("branch", "")
    degree = parsed.get("academic", {}).get("degree", "")
    grad_year = parsed.get("academic", {}).get("yearOfGraduation", "")

    text = (
        f"Skills: {', '.join(tech)}; "
        f"CGPA: {cgpa}; "
        f"Branch: {branch}; "
        f"Degree: {degree}; "
        f"Graduation Year: {grad_year}"
    )

    embedding = embed_model.encode(text).tolist()
    embedding = normalize(embedding)

    return embedding

# ---------- API: UPLOAD RESUME ----------
@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    body = await file.read()

    # Extract text
    try:
        text = extract_resume_text_from_bytes(body)
        if not text:
            raise HTTPException(status_code=400, detail="PDF had no extractable text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {e}")

    # Gemini parsing
    try:
        parsed = parse_resume_with_gemini(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {e}")

    # Metadata
    parsed.setdefault("meta", {})
    parsed["meta"]["createdAt"] = datetime.datetime.utcnow().isoformat() + "Z"
    parsed["resume"]["uploadedAt"] = datetime.datetime.utcnow().isoformat() + "Z"

    # Store in Firestore
    try:
        doc_id = store_student(db, parsed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firestore error: {e}")

    # ---------- NEW: Store embedding in Pinecone ----------
    try:
        emb = build_embedding_payload(parsed)

        pinecone_index.upsert([
            {
                "id": doc_id,
                "values": emb,
                "metadata": {
                    "name": parsed.get("name", ""),
                    "email": parsed.get("email", ""),
                    "branch": parsed.get("academic", {}).get("branch", "")
                }
            }
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone error: {e}")

    return JSONResponse(status_code=201, content={"doc_id": doc_id, "message": "Stored & Embedded"})

# ---------- HEALTH ----------
@app.get("/")
def root():
    return {"status": "ok"}

# =========================================================
# 🔥 JD → TEXT (skills + preferred skills + description)
# =========================================================

def jd_to_text(jd):
    must = jd.get("mustHaveSkills", [])
    good = jd.get("goodToHaveSkills", [])
    desc = jd.get("description", "")

    text = (
        f"Required Skills: {', '.join(must)}; "
        f"Preferred Skills: {', '.join(good)}; "
        f"Job Description: {desc}"
    )
    return text


# =========================================================
# 🔥 STUDENT → TEXT (technical skills + tools + project descriptions)
# =========================================================

def student_to_text(student):
    skills = student.get("skills", {}).get("technical", [])
    tools = student.get("skills", {}).get("tools", [])
    projects = student.get("projects", [])

    proj_desc = []
    for p in projects:
        proj_desc.append(f"{p.get('name','')}: {p.get('description','')}")

    text = (
        f"Skills: {', '.join(skills)}; "
        f"Tools: {', '.join(tools)}; "
        f"Projects: {' | '.join(proj_desc)}"
    )
    return text


# =========================================================
# 🔥 COSINE SIMILARITY
# =========================================================

import numpy as np

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# =========================================================
# 🔥 EMAIL SENDER (Free — Gmail SMTP)
# =========================================================

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(to_email: str, subject: str, message_text: str):

    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print("SMTP Email/Password missing — cannot send email")
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(message_text, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)

        print(f"Email sent to {to_email}")

    except Exception as e:
        print("EMAIL ERROR:", e)


# =========================================================
# 🔥 FINAL ELIGIBILITY ENDPOINT (NO GEMINI, MINIMAL OUTPUT)
# =========================================================

@app.post("/check-eligibility")
async def eligibility_check(payload: Dict[str, Any]):

    company = payload.get("company")
    jd = payload.get("jd")

    if not company:
        raise HTTPException(status_code=400, detail="Company name required")

    if not jd:
        raise HTTPException(status_code=400, detail="JD JSON required")

    print("\n===== ELIGIBILITY CHECK STARTED =====\n")

    # 1️⃣ JD → text
    jd_text = jd_to_text(jd)
    print("DEBUG JD TEXT:", jd_text)

    # 2️⃣ Embed JD
    jd_embedding = embed_model.encode(jd_text).tolist()
    jd_embedding = normalize(jd_embedding)

    # 3️⃣ Query Pinecone
    try:
        results = pinecone_index.query(
            vector=jd_embedding,
            top_k=50,
            include_metadata=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

    print("DEBUG: Pinecone returned matches =", len(results.matches))

    eligible_students = []
    THRESHOLD = 0.45

    # 4️⃣ Loop over students
    for match in results.matches:
        sid = match.id
        print("CHECKING STUDENT:", sid)

        doc = db.collection("student").document(sid).get()
        if not doc.exists:
            print("Student not found in Firestore:", sid)
            continue

        student = doc.to_dict()

        stu_text = student_to_text(student)
        stu_emb = embed_model.encode(stu_text).tolist()
        stu_emb = normalize(stu_emb)

        sim = cosine(jd_embedding, stu_emb)
        print(f"SIMILARITY_SCORE({sid}) =", sim)

        if sim >= THRESHOLD:
            name = student.get("name", "")
            email = student.get("email", "")

            # save minimal student info
            eligible_students.append({
                "name": name,
                "email": email,
                "similarity": float(sim)
            })

            # 🔥 SEND EMAIL NOW
            send_email(
                email,
                f"You are eligible for {company}",
                f"Hi {name},\n\n"
                f"Great news! Your resume closely matches the requirements for {company}. "
                f"The job description aligns well with your skills, projects, and experience.\n\n"
                f"This means you have a strong chance of being shortlisted further.\n\n"
                f"Please log in to Plac1fy and apply right away to secure your opportunity.\n\n"
                f"Best of luck!\n"
                f"– Placement Team"
            )

        else:
            print("Skipping due to low similarity:", sim)

    # 5️⃣ Save results to Firestore
    db.collection("eligibility_results").document(company).set({
        "company": company,
        "jd": jd,
        "eligibleStudents": eligible_students,
        "count": len(eligible_students),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    })

    print("\n===== ELIGIBILITY CHECK COMPLETE =====\n")

    # 6️⃣ Return minimal output
    return {
        "company": company,
        "eligibleCount": len(eligible_students),
        "eligibleStudents": eligible_students
    }


# =========================================================
# 🔥 DEBUGGING (PINECONE STATS)
# =========================================================

stats = pinecone_index.describe_index_stats()
print("DEBUG: Pinecone index stats =", stats)
