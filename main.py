from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import os, cv2, base64
from insightface.app import FaceAnalysis
import traceback


app = FastAPI()

# -------------------- CONFIG --------------------
DET_SIZE = 320
DET_SIZE_RECO = 640
MATCH_THRESHOLD = 0.4
BASE_DB_DIR = "user_dbs"  # each user will have their own DB folder

os.makedirs(BASE_DB_DIR, exist_ok=True)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))

app_insight = FaceAnalysis(name="buffalo_l")
app_insight.prepare(ctx_id=-1, det_size=(DET_SIZE_RECO, DET_SIZE_RECO))

# -------------------- HELPERS --------------------
def get_user_db_path(email: str):
    """Get per-user DB path."""
    safe_email = email.replace("@", "_").replace(".", "_")
    return os.path.join(BASE_DB_DIR, f"{safe_email}_db.npz")

def l2_normalize(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def save_database(labels, embeddings, db_path):
    np.savez_compressed(db_path, labels=np.array(labels, dtype=object), embeddings=embeddings.astype(np.float32))

def load_database(db_path):
    if not os.path.exists(db_path):
        return [], np.empty((0, 512), dtype=np.float32)
    data = np.load(db_path, allow_pickle=True)
    return data["labels"].tolist(), data["embeddings"].astype(np.float32)

def match_face(embedding, embeddings_db, labels, threshold):
    if embeddings_db.size == 0:
        return "Unknown", 0.0
    sims = np.dot(embeddings_db, embedding)
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        return labels[idx], sims[idx]
    return "Unknown", sims[idx]

def annotate_faces(img, faces, labels, embeddings_db, threshold):
    recognized_ids = []
    unrec_img = img.copy()
    for face in faces:
        emb = l2_normalize(face.embedding)
        name, score = match_face(emb, embeddings_db, labels, threshold)
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{name} ({score:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if name == "Unknown":
            cv2.rectangle(unrec_img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(unrec_img, f"{name} ({score:.2f})", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            recognized_ids.append(name)
    return recognized_ids, img, unrec_img

def img_to_base64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

# -------------------- ROUTES --------------------

@app.get("/")
def root():
    return {"hello world"}

@app.post("/signup")
def signup(email: str = Form(...)):
    try:
        db_path = get_user_db_path(email)
        if not os.path.exists(db_path):
            save_database([], np.empty((0, 512), dtype=np.float32), db_path)
            return {"message": f"New DB created for {email}"}
        return {"message": f"DB already exists for {email}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signup failed: {str(e)}")

@app.post("/register")
async def register_student(
    enroll_no: str = Form(...),
    images: List[UploadFile] = None,
    userEmail: str = Header(...)
):
    try:
        db_path = get_user_db_path(userEmail)
        labels, embeddings_db = load_database(db_path)

        if not images or len(images) != 3:
            raise HTTPException(status_code=400, detail="Please upload exactly 3 images.")

        embeddings = []
        for img_file in images:
            img_bytes = await img_file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            faces = face_app.get(img)
            if not faces:
                raise HTTPException(status_code=400, detail=f"No face detected in {img_file.filename}")

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings.append(l2_normalize(face.embedding))

        mean_emb = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

        if enroll_no in labels:
            idx = labels.index(enroll_no)
            embeddings_db[idx] = mean_emb
            msg = f"Updated entry for {enroll_no}"
        else:
            labels.append(enroll_no)
            embeddings_db = np.vstack([embeddings_db, mean_emb]) if embeddings_db.size else mean_emb[np.newaxis, :]
            msg = f"Registered new student {enroll_no}"

        save_database(labels, embeddings_db, db_path)
        return {"message": msg, "total_students": len(labels)}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Register failed: {str(e)}")

@app.post("/recognize")
async def recognize_classroom(
    file: UploadFile = File(...),
    userEmail: str = Header(...)
):
    try:
        db_path = get_user_db_path(userEmail)
        labels, embeddings_db = load_database(db_path)

        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        faces = app_insight.get(img)
        rec_ids, all_faces_img, unrec_img = annotate_faces(
            img.copy(), faces, labels, embeddings_db, MATCH_THRESHOLD
        )

        result = {
            "recognized_ids": rec_ids,
            "annotated_all": img_to_base64(all_faces_img),
            "annotated_unrecognized": img_to_base64(unrec_img)
        }
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
