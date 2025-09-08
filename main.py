from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import os, cv2, base64
from insightface.app import FaceAnalysis
import traceback
from PIL import Image
import pillow_heif  # ensures HEIC support



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

# Common function used by both routes ie. reco and deReco

async def process_images(files: List[UploadFile], userEmail: str):
    """
    Process uploaded files and return:
      - recognized_students: List[dict]
      - annotated_all_imgs: List[np.ndarray]
      - annotated_unrec_imgs: List[np.ndarray]
    Handles JPEG/PNG via OpenCV, and falls back to HEIC via pillow_heif+Pillow.
    Skips invalid/unreadable files and raises 400 if none valid.
    """
    db_path = get_user_db_path(userEmail)
    labels, embeddings_db = load_database(db_path)

    all_recognized = []
    annotated_all_imgs = []
    annotated_unrec_imgs = []

    any_valid = False
    invalid_files = []

    for file in files:
        filename = getattr(file, "filename", "<unknown>")
        try:
            img_bytes = await file.read()
            if not img_bytes:
                invalid_files.append(filename)
                continue

            # Try OpenCV first (fast; supports jpg/png)
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            # If OpenCV failed (e.g. HEIC), try pillow_heif -> PIL -> OpenCV
            if img is None:
                try:
                    heif_file = pillow_heif.read_heif(img_bytes)
                    # Create a PIL Image from the heif data
                    pil_img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    # Convert PIL to OpenCV BGR
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    # Last-ditch: try Pillow to open common formats (some streams)
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None

            if img is None:
                invalid_files.append(filename)
                continue

            any_valid = True

            # Run face detection & recognition on the decoded image
            try:
                faces = app_insight.get(img)
            except Exception:
                # Detector failed for this image; log and skip this file
                traceback.print_exc()
                invalid_files.append(filename)
                continue

            rec_ids, all_faces_img, unrec_img = annotate_faces(
                img.copy(), faces, labels, embeddings_db, MATCH_THRESHOLD
            )

            # Collect recognized students
            for rid in rec_ids:
                if ":" in rid:
                    roll_no, name = rid.split(":", 1)
                    all_recognized.append({"roll_no": roll_no, "name": name})
                else:
                    all_recognized.append({"roll_no": rid, "name": None})

            # Save annotated images for merging (only add if returned)
            if all_faces_img is not None:
                annotated_all_imgs.append(all_faces_img)
            if unrec_img is not None:
                annotated_unrec_imgs.append(unrec_img)

        except Exception:
            traceback.print_exc()
            invalid_files.append(filename)
            continue

    if not any_valid:
        detail = "No valid images received"
        if invalid_files:
            detail += f"; invalid files: {', '.join(invalid_files[:10])}"
        raise HTTPException(status_code=400, detail=detail)

    # Deduplicate recognized students by (roll_no, name)
    unique_students = {
        (student["roll_no"], student["name"]): student
        for student in all_recognized
    }
    recognized_students = list(unique_students.values())

    return recognized_students, annotated_all_imgs, annotated_unrec_imgs


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
    Rollno: str = Form(...),
    studentName: str = Form(...),
    images: List[UploadFile] = File(...),   # required; client can send 1..3 files
    userEmail: str = Header(...)
):
    try:
        # validate number of files: at least 1, at most 3
        if not images or len(images) < 1 or len(images) > 3:
            raise HTTPException(status_code=400, detail="Please upload between 1 and 3 images.")

        db_path = get_user_db_path(userEmail)
        labels, embeddings_db = load_database(db_path)

        embeddings = []
        for img_file in images:
            img_bytes = await img_file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            faces = face_app.get(img)
            if not faces:
                raise HTTPException(status_code=400, detail=f"No face detected in {img_file.filename}")

            # choose the largest face
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings.append(l2_normalize(face.embedding))

        # mean of available embeddings (works for 1..3 images)
        mean_emb = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))

        # Store tuple (roll_no, name) as label
        student_id = f"{Rollno}:{studentName}"

        if any(lbl.startswith(Rollno + ":") for lbl in labels):
            # Update existing entry for this roll number (also update name)
            idx = next(i for i, lbl in enumerate(labels) if lbl.startswith(Rollno + ":"))
            labels[idx] = student_id
            embeddings_db[idx] = mean_emb
            msg = f"Updated entry for {Rollno} ({studentName})"
        else:
            labels.append(student_id)
            embeddings_db = np.vstack([embeddings_db, mean_emb]) if embeddings_db.size else mean_emb[np.newaxis, :]
            msg = f"Registered new student {Rollno} ({studentName})"

        save_database(labels, embeddings_db, db_path)
        return {
            "message": msg,
            "total_students": len(labels),
            "student": {"roll_no": Rollno, "name": studentName}
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Register failed: {str(e)}")

    

@app.post("/recognize")
async def recognize_classroom(
    files: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    try:
        recognized_students, _, _ = await process_images(files, userEmail)

        result = {
            "recognized_students": recognized_students
        }
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")


# 📌 Debug recognition route (returns annotated images too)
@app.post("/debugRecognize")
async def debug_recognize_classroom(
    files: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    try:
        recognized_students, annotated_all_imgs, annotated_unrec_imgs = await process_images(files, userEmail)

        # Merge annotated images into contact sheets
        def merge_images(img_list):
            if not img_list:
                return None
            heights = [img.shape[0] for img in img_list]
            target_h = min(heights)
            resized = [
                cv2.resize(img, (int(img.shape[1] * target_h / img.shape[0]), target_h))
                for img in img_list
            ]
            return cv2.hconcat(resized)

        merged_all = merge_images(annotated_all_imgs)
        merged_unrec = merge_images(annotated_unrec_imgs)

        result = {
            "recognized_students": recognized_students,
            "annotated_all": img_to_base64(merged_all) if merged_all is not None else None,
            "annotated_unrecognized": img_to_base64(merged_unrec) if merged_unrec is not None else None
        }
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Debug recognition failed: {str(e)}")

    
@app.get("/students")
def list_students(userEmail: str = Header(...)):
    try:
        db_path = get_user_db_path(userEmail)
        labels, _ = load_database(db_path)

        if not labels:
            return {"message": f"No students registered for {userEmail}", "students": []}

        students = []
        for lbl in labels:
            if ":" in lbl:
                roll_no, name = lbl.split(":", 1)
                students.append({"roll_no": roll_no, "name": name})
            else:
                students.append({"roll_no": lbl, "name": None})

        return {
            "message": f"Total {len(students)} students found for {userEmail}",
            "students": students
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch students: {str(e)}")
    


@app.delete("/deleteStudent")
def delete_student(
    Rollno: str = Form(...),
    studentName: str = Form(...),
    userEmail: str = Header(...)
):
    try:
        db_path = get_user_db_path(userEmail)
        labels, embeddings_db = load_database(db_path)

        target_label = f"{Rollno}:{studentName}"

        # find all indices that exactly match the target label
        indices = [i for i, lbl in enumerate(labels) if lbl == target_label]
        if not indices:
            raise HTTPException(status_code=404, detail=f"Student {target_label} not found for {userEmail}.")

        # Remove labels at indices (do it in reverse to avoid reindexing issues)
        for idx in sorted(indices, reverse=True):
            labels.pop(idx)
            if embeddings_db.size:
                embeddings_db = np.delete(embeddings_db, idx, axis=0)

        save_database(labels, embeddings_db, db_path)

        return {
            "message": f"Deleted {len(indices)} record(s) for {target_label}",
            "total_students": len(labels)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")




