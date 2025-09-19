from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import os, cv2, base64, io
from insightface.app import FaceAnalysis
import traceback
from PIL import Image
import pillow_heif  # ensures HEIC support
import uuid
from datetime import datetime

app = FastAPI()

# -------------------- CONFIG --------------------
DET_SIZE = 320
DET_SIZE_RECO = 1280
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

def save_database(metadata: List[dict], embeddings: np.ndarray, db_path: str):
    """
    Save metadata (as JSON string) + embeddings to npz.
    metadata: list of dicts with keys id, roll_no, name, created_at, updated_at
    """
    import json
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    # np.savez_compressed will create db_path as given (no extra .npz appended)
    np.savez_compressed(db_path, metadata=np.array(metadata_json), embeddings=embeddings.astype(np.float32))

def load_database(db_path: str):
    """
    Return (metadata_list, embeddings_array)
    Backwards-compatible: if old file stored 'labels' array of 'roll:name' strings, convert to metadata entries (new uuid assigned).
    """
    import json
    if not os.path.exists(db_path):
        return [], np.empty((0, 512), dtype=np.float32)

    data = np.load(db_path, allow_pickle=True)
    # New format: 'metadata' + 'embeddings'
    if "metadata" in data:
        metadata_json = data["metadata"].tolist()
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode("utf-8")
        metadata = json.loads(metadata_json)
        embeddings = data["embeddings"].astype(np.float32)
        return metadata, embeddings

    # Backward compatibility: old format had 'labels' + 'embeddings'
    # labels were like "roll:name"
    if "labels" in data:
        labels = data["labels"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
        # convert labels (strings) to metadata entries (generating UUIDs)
        metadata = []
        now = datetime.utcnow().isoformat() + "Z"
        for lbl in labels:
            if isinstance(lbl, bytes):
                lbl = lbl.decode("utf-8")
            if ":" in lbl:
                roll, name = lbl.split(":", 1)
            else:
                roll, name = lbl, None
            metadata.append({
                "id": str(uuid.uuid4()),
                "roll_no": roll,
                "name": name,
                "created_at": now,
                "updated_at": now
            })
        # Save converted DB immediately for future runs
        try:
            save_database(metadata, embeddings, db_path)
        except Exception:
            traceback.print_exc()
        return metadata, embeddings

    # Unknown format
    raise RuntimeError("DB file exists but lacks expected fields (metadata|labels).")

def labels_from_metadata(metadata: List[dict]):
    """
    Helper to produce list of label-strings used previously for matching/display: "roll:name"
    If name is None, returns roll only.
    """
    out = []
    for m in metadata:
        rn = m.get("roll_no", "")
        nm = m.get("name")
        if nm:
            out.append(f"{rn}:{nm}")
        else:
            out.append(rn)
    return out

def match_face(embedding, embeddings_db, labels, threshold):
    if embeddings_db.size == 0:
        return "Unknown", 0.0
    sims = np.dot(embeddings_db, embedding)
    idx = np.argmax(sims)
    if sims[idx] >= threshold:
        return labels[idx], float(sims[idx])
    return "Unknown", float(sims[idx])

def annotate_faces(img, faces, labels, embeddings_db, threshold):
    """
    labels here is the list of label-strings (roll:name or roll)
    embeddings_db aligns with labels order.
    Returns recognized_ids (list of label-strings), annotated img, unrecognized annotated img
    """
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
    if img is None:
        return None
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
    metadata, embeddings_db = load_database(db_path)
    labels = labels_from_metadata(metadata)  # keep alignment

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
                    # find matching metadata entry to attach id (first match)
                    matched = next((m for m in metadata if m.get("roll_no") == roll_no and m.get("name") == name), None)
                    if matched:
                        all_recognized.append({"id": matched["id"], "roll_no": roll_no, "name": name})
                    else:
                        all_recognized.append({"roll_no": roll_no, "name": name})
                else:
                    # roll-only labels
                    matched = next((m for m in metadata if m.get("roll_no") == rid), None)
                    if matched:
                        all_recognized.append({"id": matched["id"], "roll_no": matched["roll_no"], "name": matched.get("name")})
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

    # Deduplicate recognized students by id/roll_no,name
    unique_keyed = {}
    for s in all_recognized:
        key = (s.get("id") or s.get("roll_no"), s.get("name"))
        unique_keyed[key] = s
    recognized_students = list(unique_keyed.values())

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
    

# ---------- Replace /register with this (creation-only, rejects exact duplicate roll+name) ----------
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
        metadata, embeddings_db = load_database(db_path)
        labels = labels_from_metadata(metadata)

        # If exact (roll_no, name) pair already present -> reject as duplicate
        duplicate = next((m for m in metadata if m.get("roll_no") == Rollno and (m.get("name") or "") == studentName), None)
        if duplicate is not None:
            raise HTTPException(status_code=400, detail=f"Student already registered with id {duplicate['id']}")

        # proceed to extract embeddings from images (same as before)
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
        mean_emb = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0)).astype(np.float32)

        # Create new student entry with UUID (do NOT update existing entries here)
        student_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        new_record = {
            "id": student_id,
            "roll_no": Rollno,
            "name": studentName,
            "created_at": now,
            "updated_at": now
        }
        metadata.append(new_record)
        if embeddings_db.size:
            embeddings_db = np.vstack([embeddings_db, mean_emb])
        else:
            embeddings_db = mean_emb[np.newaxis, :]

        save_database(metadata, embeddings_db, db_path)
        msg = f"Registered new student {Rollno} ({studentName}) with id {student_id}"

        return {
            "message": msg,
            "total_students": len(metadata),
            "student": {"id": student_id, "roll_no": Rollno, "name": studentName}
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
        metadata, _ = load_database(db_path)

        if not metadata:
            return {"message": f"No students registered for {userEmail}", "students": []}

        students = []
        for m in metadata:
            students.append({"id": m.get("id"), "roll_no": m.get("roll_no"), "name": m.get("name")})

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
    """
    Backwards-compatible delete by (Rollno, studentName) pair.
    """
    try:
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        target_roll = Rollno
        target_name = studentName

        # find all indices that match both roll and name
        indices = [i for i, m in enumerate(metadata) if m.get("roll_no") == target_roll and m.get("name") == target_name]
        if not indices:
            raise HTTPException(status_code=404, detail=f"Student {target_roll}:{target_name} not found for {userEmail}.")

        # Remove metadata and corresponding embeddings (reverse order)
        for idx in sorted(indices, reverse=True):
            metadata.pop(idx)
            if embeddings_db.size:
                embeddings_db = np.delete(embeddings_db, idx, axis=0)

        save_database(metadata, embeddings_db, db_path)

        return {
            "message": f"Deleted {len(indices)} record(s) for {target_roll}:{target_name}",
            "total_students": len(metadata)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.delete("/studentDelete")
def delete_student_by_id(student_id: str = Form(...), userEmail: str = Header(...)):
    """
    Preferred delete-by-UUID endpoint.
    """
    try:
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        metadata.pop(idx)
        if embeddings_db.size:
            embeddings_db = np.delete(embeddings_db, idx, axis=0)

        save_database(metadata, embeddings_db, db_path)
        return {"message": "Deleted", "total_students": len(metadata)}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

# ---------- Add this PUT route (update metadata by UUID) ----------
@app.put("/studentsUpdate")
def update_student_metadata(
    student_id: str = Form(...),
    roll_no: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    userEmail: str = Header(...)
):
    """
    Update student's roll_no and/or name by UUID.
    - If the requested new (roll_no, name) pair would exactly match another student, returns 400.
    - Fields are optional; unspecified fields keep their current value.
    """
    try:
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        # Determine proposed new values (use existing if not provided)
        current = metadata[idx]
        new_roll = roll_no if roll_no is not None else current.get("roll_no")
        new_name = name if name is not None else current.get("name")

        # Check collision: is there any OTHER student with exactly same roll+name?
        coll = next((m for m in metadata if m.get("id") != student_id and m.get("roll_no") == new_roll and (m.get("name") or "") == (new_name or "")), None)
        if coll is not None:
            # Return a clear error listing the conflicting student's id
            raise HTTPException(
                status_code=400,
                detail=f"Update would collide with existing student id {coll['id']} (roll_no={coll.get('roll_no')}, name={coll.get('name')}). Choose a different roll_no/name."
            )

        # No collision — perform update
        metadata[idx]["roll_no"] = new_roll
        metadata[idx]["name"] = new_name
        metadata[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_database(metadata, embeddings_db, db_path)
        return {"message": "Updated", "student": metadata[idx]}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/students/re-enroll")
async def reenroll_student_embeddings(
    student_id: str = Form(...),
    images: List[UploadFile] = File(...),  # 1..3 images
    userEmail: str = Header(...)
):
    """
    Replace/update embeddings for an existing student (by UUID).
    Accepts 1..3 face images, computes mean embedding and replaces the student's embedding.
    """
    try:
        # validate number of files
        if not images or len(images) < 1 or len(images) > 3:
            raise HTTPException(status_code=400, detail="Please upload between 1 and 3 images for re-enrollment.")

        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        embeddings_list = []
        invalid_files = []
        for img_file in images:
            filename = getattr(img_file, "filename", "<unknown>")
            try:
                img_bytes = await img_file.read()
                if not img_bytes:
                    invalid_files.append(filename)
                    continue

                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if img is None:
                    # try fallback HEIC/Pillow path if you want (optional)
                    invalid_files.append(filename)
                    continue

                faces = face_app.get(img)
                if not faces:
                    invalid_files.append(filename)
                    continue

                # choose largest face and normalize
                face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                embeddings_list.append(l2_normalize(face.embedding))

            except Exception:
                traceback.print_exc()
                invalid_files.append(filename)
                continue

        if not embeddings_list:
            detail = "No valid faces found in provided images."
            if invalid_files:
                detail += f" Invalid files: {', '.join(invalid_files[:10])}."
            raise HTTPException(status_code=400, detail=detail)

        # mean and normalize
        mean_emb = l2_normalize(np.mean(np.stack(embeddings_list, axis=0), axis=0)).astype(np.float32)

        # Ensure embeddings_db has same number of rows as metadata.
        # If embeddings_db is empty or length mismatch, pad/trim with zeros so indices align.
        emb_dim = mean_emb.shape[0]
        if embeddings_db.size == 0:
            embeddings_db = np.zeros((len(metadata), emb_dim), dtype=np.float32)
        elif embeddings_db.shape[0] != len(metadata):
            # if rows fewer -> pad; if more -> trim (shouldn't usually happen)
            new_db = np.zeros((len(metadata), emb_dim), dtype=np.float32)
            rows_to_copy = min(embeddings_db.shape[0], len(metadata))
            new_db[:rows_to_copy, :min(embeddings_db.shape[1], emb_dim)] = embeddings_db[:rows_to_copy, :min(embeddings_db.shape[1], emb_dim)]
            embeddings_db = new_db

        # replace the embedding row
        embeddings_db[idx] = mean_emb

        # update metadata timestamp
        metadata[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_database(metadata, embeddings_db, db_path)

        return {
            "message": "Re-enrollment successful. Embeddings updated.",
            "student": {"id": student_id, "roll_no": metadata[idx].get("roll_no"), "name": metadata[idx].get("name")},
            "total_students": len(metadata)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Re-enroll failed: {str(e)}")


