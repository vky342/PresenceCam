from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Union, Tuple
import numpy as np
import json
import os, cv2, base64, io
from insightface.app import FaceAnalysis
import traceback
from PIL import Image
import pillow_heif  # ensures HEIC support
import uuid
from datetime import datetime
from pathlib import Path
import imghdr
import shutil
import uuid as uuidlib
from fastapi.responses import Response

app = FastAPI()

# -------------------- Profile Picture Storing --------------------

try:
    from fastapi import UploadFile
except Exception:
    UploadFile = object  # fallback for environments without FastAPI

ImageLike = Union[UploadFile, bytes, bytearray, memoryview]


def create_directory(base_dir: Union[str, Path]) -> Path:
    """
    Create base directory if it doesn't exist and return Path.
    """
    p = Path(base_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _guess_ext_from_bytes(b: bytes) -> str:
    """
    Return extension starting with '.' (e.g. '.png') guessed by imghdr,
    or empty string if unknown.
    """
    kind = imghdr.what(None, h=b)
    if not kind:
        return ""
    # imghdr returns e.g. 'jpeg', 'png', 'gif'
    if kind == "jpeg":
        return ".jpg"
    return f".{kind}"


def _get_bytes_and_ext(item: ImageLike, fallback_ext: str = ".jpg") -> Tuple[bytes, str]:
    """
    Normalize an image-like item to (bytes, extension).
    For UploadFile: read .filename for extension; if missing, guess from bytes.
    For bytes-like: guess extension from bytes; fallback to fallback_ext.
    """
    # UploadFile-like
    if hasattr(item, "read") and hasattr(item, "filename"):
        # It's probably a FastAPI UploadFile or similar
        # IMPORTANT: we expect item to already be at a readable position; for UploadFile passed
        # directly from endpoints we usually read it asynchronously before calling this.
        # For safety, try reading .file then .read()
        raw = None
        try:
            if hasattr(item, "file") and hasattr(item.file, "read"):
                raw = item.file.read()
        except Exception:
            raw = None
        if raw is None:
            try:
                raw = item.read()
            except Exception:
                raw = None

        if raw is None:
            raise ValueError("Could not read bytes from UploadFile-like object")

        filename = getattr(item, "filename", "") or ""
        suffix = Path(filename).suffix.lower()
        if suffix:
            return raw, suffix
        guessed = _guess_ext_from_bytes(raw)
        return raw, guessed or fallback_ext

    # bytes-like
    if isinstance(item, (bytes, bytearray, memoryview)):
        raw = bytes(item)
        guessed = _guess_ext_from_bytes(raw)
        return raw, guessed or fallback_ext

    # Unknown type
    raise TypeError(f"Unsupported image type: {type(item)}")


def save_images(uuid_str: str, images: List[ImageLike], base_dir: Union[str, Path]) -> List[Path]:
    """
    Save a list of images for the given UUID into base_dir/uuid_str/.
    Filenames will be: {uuid_str}_{index}_{randomhex}{ext}
    Returns list of Path objects to saved files.
    """
    base = create_directory(base_dir)
    # use a subdir per UUID
    uuid_dir = base / uuid_str
    uuid_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for i, item in enumerate(images, start=1):
        data, ext = _get_bytes_and_ext(item)
        # ensure unique filename (index plus random suffix to avoid collisions)
        unique_suffix = uuidlib.uuid4().hex[:8]
        fname = f"{uuid_str}_{i}_{unique_suffix}{ext}"
        path = uuid_dir / fname
        with path.open("wb") as f:
            f.write(data)
        saved_paths.append(path)

    return saved_paths


def get_image_paths(uuid_str: str, base_dir: Union[str, Path]) -> List[Path]:
    """
    Return list of image Paths stored for this UUID (sorted).
    """
    base = Path(base_dir)
    uuid_dir = base / uuid_str
    if not uuid_dir.exists() or not uuid_dir.is_dir():
        return []
    files = [p for p in uuid_dir.iterdir() if p.is_file()]
    files.sort()
    return files


def get_images_bytes(uuid_str: str, base_dir: Union[str, Path]) -> List[bytes]:
    """
    Return list of bytes for all images for uuid_str. Order matches get_image_paths.
    """
    paths = get_image_paths(uuid_str, base_dir)
    out = []
    for p in paths:
        out.append(p.read_bytes())
    return out


def replace_images(uuid_str: str, new_images: List[ImageLike], base_dir: Union[str, Path]) -> List[Path]:
    """
    Replace images for uuid_str with new_images. Implementation:
      - write new images to a temporary subdirectory
      - if all succeed, delete old uuid directory and move tmp to final name (atomic-ish)
    Returns list of saved Paths (in final UUID dir).
    """
    base = create_directory(base_dir)
    uuid_dir = base / uuid_str
    tmp_dir = base / f".{uuid_str}_tmp"
    # clean any stale tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        saved_tmp: List[Path] = []
        for i, item in enumerate(new_images, start=1):
            data, ext = _get_bytes_and_ext(item)
            unique_suffix = uuidlib.uuid4().hex[:8]
            fname = f"{uuid_str}_{i}_{unique_suffix}{ext}"
            path = tmp_dir / fname
            with path.open("wb") as f:
                f.write(data)
            saved_tmp.append(path)

        # all saved successfully -> remove old directory (backup) and move tmp to final
        backup_dir = None
        if uuid_dir.exists():
            backup_dir = base / f".{uuid_str}_backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            uuid_dir.rename(backup_dir)

        # move tmp to final
        tmp_dir.rename(uuid_dir)

        # cleanup backup
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir)

        # return saved final paths
        final_paths = get_image_paths(uuid_str, base_dir)
        return final_paths

    except Exception:
        # On any failure, try to clean tmp and leave original as-is (or restore backup)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        # Try to restore backup if exists
        possible_backup = base / f".{uuid_str}_backup"
        if possible_backup.exists():
            if uuid_dir.exists():
                shutil.rmtree(uuid_dir)
            possible_backup.rename(uuid_dir)
        raise  # re-raise to make failure visible to caller


                                            # -------------------- CONFIG --------------------
DET_SIZE = 320
DET_SIZE_RECO = 1280
MATCH_THRESHOLD = 0.4
BASE_DB_DIR = "user_dbs"  # each user will have their own DB folder
IMAGES_DIR = Path("stored_images")  # root for images; you can change this

MAX_IMAGES = 3
MIN_IMAGES = 1

os.makedirs(BASE_DB_DIR, exist_ok=True)
create_directory(IMAGES_DIR)

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
    if "labels" in data:
        labels = data["labels"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
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
        try:
            save_database(metadata, embeddings, db_path)
        except Exception:
            traceback.print_exc()
        return metadata, embeddings

    raise RuntimeError("DB file exists but lacks expected fields (metadata|labels).")


def labels_from_metadata(metadata: List[dict]):
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
async def process_images(files: List[UploadFile], userEmail: str, classId: str):
    db_path = get_user_db_path(userEmail)
    metadata, embeddings_db = load_database(db_path)
    metadata, embeddings_db = filter_by_class(metadata, embeddings_db, classId)
    labels = labels_from_metadata(metadata)

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

            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img is None:
                try:
                    heif_file = pillow_heif.read_heif(img_bytes)
                    pil_img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    try:
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None

            if img is None:
                invalid_files.append(filename)
                continue

            any_valid = True

            try:
                faces = app_insight.get(img)
            except Exception:
                traceback.print_exc()
                invalid_files.append(filename)
                continue

            rec_ids, all_faces_img, unrec_img = annotate_faces(
                img.copy(), faces, labels, embeddings_db, MATCH_THRESHOLD
            )

            for rid in rec_ids:
                if ":" in rid:
                    roll_no, name = rid.split(":", 1)
                    matched = next((m for m in metadata if m.get("roll_no") == roll_no and m.get("name") == name), None)
                    if matched:
                        all_recognized.append({"id": matched["id"], "roll_no": roll_no, "name": name})
                    else:
                        all_recognized.append({"roll_no": roll_no, "name": name})
                else:
                    matched = next((m for m in metadata if m.get("roll_no") == rid), None)
                    if matched:
                        all_recognized.append({"id": matched["id"], "roll_no": matched["roll_no"], "name": matched.get("name")})
                    else:
                        all_recognized.append({"roll_no": rid, "name": None})

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

    unique_keyed = {}
    for s in all_recognized:
        key = (s.get("id") or s.get("roll_no"), s.get("name"))
        unique_keyed[key] = s
    recognized_students = list(unique_keyed.values())

    return recognized_students, annotated_all_imgs, annotated_unrec_imgs


def filter_by_class(metadata, embeddings_db, class_id):
    """
    Returns (filtered_metadata, filtered_embeddings)
    preserving index alignment.
    """
    idxs = [i for i, m in enumerate(metadata) if m.get("class") == class_id]

    if not idxs:
        return [], np.empty((0, embeddings_db.shape[1] if embeddings_db.size else 512), dtype=np.float32)

    filtered_meta = [metadata[i] for i in idxs]
    filtered_emb = embeddings_db[idxs] if embeddings_db.size else np.empty((0, 512), dtype=np.float32)

    return filtered_meta, filtered_emb

def get_user_classes_path(email: str):
    safe_email = email.replace("@", "_").replace(".", "_")
    return os.path.join(BASE_DB_DIR, f"{safe_email}_classes.json")


def load_classes(email: str):
    path = get_user_classes_path(email)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_classes(email: str, classes: list):
    path = get_user_classes_path(email)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)


def get_class_by_id(classes, class_id: str):
    return next((c for c in classes if c["id"] == class_id), None)


def get_class_by_name(classes, name: str):
    return next((c for c in classes if c["name"].lower() == name.lower()), None)



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


# ---------- Register (now saves images too, with rollback on failure) ----------
@app.post("/register")
async def register_student(
    Rollno: str = Form(...),
    studentName: str = Form(...),
    classId: str = Form(...),
    images: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    try:
        # validate number of files: at least 1, at most 3
        if not images or len(images) < MIN_IMAGES or len(images) > MAX_IMAGES:
            raise HTTPException(status_code=400, detail=f"Please upload between {MIN_IMAGES} and {MAX_IMAGES} images.")

        db_path = get_user_db_path(userEmail)
        classes = load_classes(userEmail)
        if not get_class_by_id(classes, classId):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid classId. Please select or create a class first."
                    )
        
        metadata, embeddings_db = load_database(db_path)
        labels = labels_from_metadata(metadata)

        # If exact (roll_no, name) pair already present -> reject as duplicate
        duplicate = next((m for m in metadata if m.get("class") == classId and m.get("roll_no") == Rollno and (m.get("name") or "") == studentName),None)
        if duplicate is not None:
            raise HTTPException(status_code=400, detail=f"Student already registered with id {duplicate['id']}")

        # READ images once into memory (we'll reuse bytes for embedding extraction AND saving)
        raw_images_bytes: List[bytes] = []
        for img_file in images:
            content = await img_file.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Empty file: {getattr(img_file, 'filename', '<unknown>')}")
            raw_images_bytes.append(content)

        # proceed to extract embeddings from images using bytes we have
        embeddings = []
        invalid_count = 0
        for idx, img_bytes in enumerate(raw_images_bytes):
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                # attempt HEIC/PIL fallback quickly
                try:
                    heif_file = pillow_heif.read_heif(img_bytes)
                    pil_img = Image.frombytes(
                        heif_file.mode,
                        heif_file.size,
                        heif_file.data,
                        "raw",
                        heif_file.mode,
                        heif_file.stride,
                    )
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    img = None
            if img is None:
                raise HTTPException(status_code=400, detail=f"Could not decode image #{idx+1}")

            faces = face_app.get(img)
            if not faces:
                raise HTTPException(status_code=400, detail=f"No face detected in uploaded image #{idx+1}")

            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embeddings.append(l2_normalize(face.embedding))

        # mean of available embeddings (works for 1..3 images)
        mean_emb = l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0)).astype(np.float32)

        # Prepare to append to DB; keep previous copy for rollback if needed
        prev_metadata = list(metadata)
        prev_embeddings_db = embeddings_db.copy() if getattr(embeddings_db, "size", 0) else embeddings_db

        # Create new student entry with UUID (do NOT update existing entries here)
        student_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        new_record = {
            "id": student_id,
            "roll_no": Rollno,
            "name": studentName,
            "class": classId,
            "created_at": now,
            "updated_at": now
            }

        metadata.append(new_record)
        if getattr(embeddings_db, "size", 0):
            embeddings_db = np.vstack([embeddings_db, mean_emb])
        else:
            embeddings_db = mean_emb[np.newaxis, :]

        # Save DB first
        save_database(metadata, embeddings_db, db_path)

        # Now attempt to save images to filesystem under IMAGES_DIR/<student_id>/
        try:
            saved_paths = save_images(student_id, raw_images_bytes, IMAGES_DIR)
        except Exception as ex_save:
            # rollback DB
            try:
                metadata[:] = prev_metadata
                embeddings_db = prev_embeddings_db
                save_database(metadata, embeddings_db, db_path)
            except Exception as ex_rollback:
                traceback.print_exc()
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save images ({ex_save}) and rollback failed ({ex_rollback}). Manual cleanup required."
                )
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to save images: {str(ex_save)}")

        msg = f"Registered new student {Rollno} ({studentName}) with id {student_id}"

        return {
            "message": msg,
            "total_students": len(metadata),
            "student": {"id": student_id, "roll_no": Rollno, "name": studentName},
            "saved_image_paths": [str(p) for p in saved_paths]
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Register failed: {str(e)}")


@app.post("/recognize")
async def recognize_classroom(
    files: List[UploadFile] = File(...),
    classId: str = Form(...),
    userEmail: str = Header(...)
):
    try:
        classes = load_classes(userEmail)
        if not get_class_by_id(classes, classId):
            raise HTTPException(
                status_code=400,
                detail="Invalid classId"
                )

        recognized_students, _, _ = await process_images(files, userEmail, classId)

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
    userEmail: str = Header(...),
    classId: str = Form(...),
):
    try:
        recognized_students, annotated_all_imgs, annotated_unrec_imgs = await process_images(files, userEmail, classId)

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
def list_students(
    classId: str,
    userEmail: str = Header(...)
):
    try:
        db_path = get_user_db_path(userEmail)
        metadata, _ = load_database(db_path)

        if not metadata:
            return {
                "message": f"No students registered for {userEmail}",
                "students": []
            }

        # Filter students by class
        students = [
            {
                "id": m.get("id"),
                "roll_no": m.get("roll_no"),
                "name": m.get("name")
            }
            for m in metadata
            if m.get("class") == classId
        ]

        if not students:
            return {
                "message": f"No students found for class '{classId}'",
                "students": []
            }

        return {
            "message": f"Total {len(students)} students found for class '{classId}'",
            "students": students
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch students: {str(e)}"
        )



@app.delete("/studentDelete")
def delete_student_by_id(
    student_id: str = Form(...),
    classId: str = Form(...),
    userEmail: str = Header(...)
):
    """
    Delete student by UUID (class-scoped):
      - Verifies student belongs to classId
      - Removes metadata and embedding row
      - Deletes stored images for that student_id
    """
    try:
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        # 🔐 Class ownership check
        if metadata[idx].get("class") != classId:
            raise HTTPException(
                status_code=403,
                detail="Student does not belong to this class"
            )

        # Remove metadata + embedding
        metadata.pop(idx)
        if embeddings_db.size:
            embeddings_db = np.delete(embeddings_db, idx, axis=0)

        save_database(metadata, embeddings_db, db_path)

        # Delete stored images directory if it exists
        student_dir = IMAGES_DIR / student_id
        if student_dir.exists() and student_dir.is_dir():
            shutil.rmtree(student_dir)

        return {
            "message": f"Deleted student {student_id} from class '{classId}'",
            "total_students": len(metadata)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




# ---------- Update metadata ----------

@app.put("/studentsUpdate")
def update_student_metadata(
    student_id: str = Form(...),
    classId: str = Form(...),
    roll_no: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    userEmail: str = Header(...)
):
    try:
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        # 🔐 Class ownership check
        if metadata[idx].get("class") != classId:
            raise HTTPException(
                status_code=403,
                detail="Student does not belong to this class"
            )

        current = metadata[idx]
        new_roll = roll_no if roll_no is not None else current.get("roll_no")
        new_name = name if name is not None else current.get("name")

        # Collision check ONLY inside same class
        coll = next(
            (
                m for m in metadata
                if m.get("id") != student_id
                and m.get("class") == classId
                and m.get("roll_no") == new_roll
                and (m.get("name") or "") == (new_name or "")
            ),
            None
        )

        if coll is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Update would collide with existing student id {coll['id']} "
                    f"in class '{classId}'. Choose a different roll_no/name."
                )
            )

        metadata[idx]["roll_no"] = new_roll
        metadata[idx]["name"] = new_name
        metadata[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_database(metadata, embeddings_db, db_path)

        return {
            "message": "Updated",
            "student": metadata[idx]
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/students/re-enroll")
async def reenroll_and_replace_images(
    student_id: str = Form(...),
    classId: str = Form(...),
    images: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    """
    Re-enroll student (class-scoped):
      - Verifies student belongs to classId
      - Updates embedding
      - Replaces stored images atomically
    """
    try:
        if not images or len(images) < MIN_IMAGES or len(images) > MAX_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Please upload between {MIN_IMAGES} and {MAX_IMAGES} images."
            )

        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idx = next((i for i, m in enumerate(metadata) if m.get("id") == student_id), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Student not found")

        # 🔐 Class ownership check (CRITICAL)
        if metadata[idx].get("class") != classId:
            raise HTTPException(
                status_code=403,
                detail="Student does not belong to this class"
            )

        # ---------- (everything below is your existing logic, unchanged) ----------

        new_images_bytes: List[bytes] = []
        invalid_files = []

        for f in images:
            fname = getattr(f, "filename", "<unknown>")
            try:
                b = await f.read()
                if not b:
                    invalid_files.append(fname)
                    continue
                new_images_bytes.append(b)
            except Exception:
                traceback.print_exc()
                invalid_files.append(fname)

        if not new_images_bytes:
            detail = "No valid image bytes received."
            if invalid_files:
                detail += f" Invalid files: {', '.join(invalid_files[:10])}."
            raise HTTPException(status_code=400, detail=detail)

        embeddings_list = []
        invalid_faces = []

        for i, img_bytes in enumerate(new_images_bytes, start=1):
            try:
                img_np = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                if img is None:
                    try:
                        heif_file = pillow_heif.read_heif(img_bytes)
                        pil_img = Image.frombytes(
                            heif_file.mode,
                            heif_file.size,
                            heif_file.data,
                            "raw",
                            heif_file.mode,
                            heif_file.stride,
                        )
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    except Exception:
                        img = None

                if img is None:
                    invalid_faces.append(f"image#{i}")
                    continue

                faces = face_app.get(img)
                if not faces:
                    invalid_faces.append(f"image#{i}")
                    continue

                face = max(
                    faces,
                    key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
                )
                embeddings_list.append(l2_normalize(face.embedding))

            except Exception:
                traceback.print_exc()
                invalid_faces.append(f"image#{i}")

        if not embeddings_list:
            detail = "No valid faces found in provided images."
            if invalid_faces:
                detail += f" Invalid/face-missing files: {', '.join(invalid_faces[:10])}."
            raise HTTPException(status_code=400, detail=detail)

        mean_emb = l2_normalize(
            np.mean(np.stack(embeddings_list, axis=0), axis=0)
        ).astype(np.float32)

        prev_metadata = list(metadata)
        prev_embeddings_db = embeddings_db.copy() if embeddings_db.size else embeddings_db

        embeddings_db[idx] = mean_emb
        metadata[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_database(metadata, embeddings_db, db_path)

        try:
            saved_paths = replace_images(student_id, new_images_bytes, IMAGES_DIR)
        except Exception as ex_replace:
            metadata[:] = prev_metadata
            embeddings_db = prev_embeddings_db
            save_database(metadata, embeddings_db, db_path)
            raise HTTPException(
                status_code=500,
                detail=f"Image replace failed: {str(ex_replace)}"
            )

        return {
            "message": "Re-enrollment and image replace successful.",
            "student": {
                "id": student_id,
                "roll_no": metadata[idx].get("roll_no"),
                "name": metadata[idx].get("name"),
                "class": classId
            },
            "saved_image_paths": [str(p) for p in saved_paths],
            "total_students": len(metadata)
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Re-enroll-and-replace failed: {str(e)}"
        )




# -------------------- NEW: image retrieval & replace endpoints --------------------

@app.post("/images/get")
async def get_images_by_uuid(uuid: str = Form(...)):
    """
    Return the raw bytes of the *first* stored image for this UUID.
    """
    try:
        paths = get_image_paths(str(uuid), IMAGES_DIR)
        if not paths:
            raise HTTPException(status_code=404, detail=f"No images found for uuid {uuid}")

        # return the first image raw
        img_bytes = paths[0].read_bytes()
        return Response(content=img_bytes, media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to collect images: {str(e)}")


# -------------------- NEW : Class System -------------------------------------------


@app.get("/classes")
def list_classes(
    userEmail: str = Header(..., alias="User-Email")
):
    try:
        classes = load_classes(userEmail)
        return {
            "total": len(classes),
            "classes": classes
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/classes")
def create_class(
    name: str = Form(...),
    userEmail: str = Header(..., alias="User-Email")
):
    try:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")

        classes = load_classes(userEmail)

        # duplicate class name (case-insensitive)
        if get_class_by_name(classes, name):
            raise HTTPException(
                status_code=400,
                detail="Class with this name already exists"
            )

        new_class = {
            "id": f"cls_{uuid.uuid4().hex[:8]}",
            "name": name,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }

        classes.append(new_class)
        save_classes(userEmail, classes)

        return new_class

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/classes")
def update_class(
    classId: str = Form(...),
    name: str = Form(...),
    userEmail: str = Header(..., alias="User-Email")
):
    try:
        name = name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")

        classes = load_classes(userEmail)

        cls = next((c for c in classes if c["id"] == classId), None)
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")

        # duplicate name check (ignore self)
        dup = next(
            (c for c in classes if c["id"] != classId and c["name"].lower() == name.lower()),
            None
        )
        if dup:
            raise HTTPException(
                status_code=400,
                detail="Another class with this name already exists"
            )

        cls["name"] = name
        cls["updated_at"] = datetime.utcnow().isoformat() + "Z"

        save_classes(userEmail, classes)

        return cls

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/classes")
def delete_class(
    classId: str = Form(...),
    userEmail: str = Header(..., alias="User-Email")
):
    try:
        classes = load_classes(userEmail)

        cls = next((c for c in classes if c["id"] == classId), None)
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")

        # Remove class
        classes = [c for c in classes if c["id"] != classId]
        save_classes(userEmail, classes)

        # 🔥 Optional but recommended: remove students of this class
        db_path = get_user_db_path(userEmail)
        metadata, embeddings_db = load_database(db_path)

        idxs = [i for i, m in enumerate(metadata) if m.get("class") == classId]
        if idxs:
            metadata = [m for i, m in enumerate(metadata) if i not in idxs]
            embeddings_db = (
                np.delete(embeddings_db, idxs, axis=0)
                if embeddings_db.size else embeddings_db
            )
            save_database(metadata, embeddings_db, db_path)

        return {
            "message": f"Class '{cls['name']}' deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
