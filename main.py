import os
import io
import json
import uuid
import uuid as uuidlib
import shutil
import base64
import traceback
import imghdr
from typing import List, Optional, Union, Tuple, Dict, Any
from pathlib import Path
from datetime import datetime
from threading import Lock
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pillow_heif
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from insightface.app import FaceAnalysis

# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    DET_SIZE = 640
    DET_SIZE_RECO = 640
    MATCH_THRESHOLD = 0.4
    BASE_DB_DIR = Path("user_dbs")
    IMAGES_DIR = Path("stored_images")
    MAX_IMAGES = 3
    MIN_IMAGES = 1

    @classmethod
    def setup(cls):
        cls.BASE_DB_DIR.mkdir(parents=True, exist_ok=True)
        cls.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# SECTION 2: UTILS & HELPERS
# =============================================================================

class ImageUtils:
    ImageLike = Union[UploadFile, bytes, bytearray, memoryview]

    @staticmethod
    def guess_ext_from_bytes(b: bytes) -> str:
        kind = imghdr.what(None, h=b)
        if not kind:
            return ""
        return ".jpg" if kind == "jpeg" else f".{kind}"

    @staticmethod
    async def get_bytes_and_ext(item: ImageLike, fallback_ext: str = ".jpg") -> Tuple[bytes, str]:
        raw = None
        filename = ""
        
        # UploadFile handling
        if hasattr(item, "read"):
            filename = getattr(item, "filename", "") or ""
            # Try async read if possible, otherwise sync
            try:
                if hasattr(item, "read"):
                    if callable(getattr(item, "read")):
                        # Check if it's awaitable (FastAPI UploadFile)
                        import inspect
                        if inspect.iscoroutinefunction(item.read):
                            raw = await item.read()
                        else:
                            raw = item.read()
            except Exception:
                pass
            
            # Fallback for spool/file
            if raw is None and hasattr(item, "file"):
                 try:
                    raw = item.file.read()
                 except Exception:
                     pass

        elif isinstance(item, (bytes, bytearray, memoryview)):
            raw = bytes(item)
        
        if raw is None:
            raise ValueError("Could not read bytes from image object")

        suffix = Path(filename).suffix.lower() if filename else ""
        if not suffix:
            suffix = ImageUtils.guess_ext_from_bytes(raw) or fallback_ext
            
        return raw, suffix

    @staticmethod
    def decode_image(img_bytes: bytes) -> Optional[np.ndarray]:
        # 1. Try OpenCV standard decode
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        # 2. HEIC Support
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
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        # 3. PIL Fallback
        try:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        return None

    @staticmethod
    def img_to_base64(img: np.ndarray) -> Optional[str]:
        if img is None: return None
        _, buf = cv2.imencode(".jpg", img)
        return base64.b64encode(buf).decode()

    @staticmethod
    def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


# =============================================================================
# SECTION 3: CORE SERVICES (Storage, Database, Face)
# =============================================================================

class StorageService:
    @staticmethod
    def save_images(uuid_str: str, images_bytes_list: List[bytes], exts_list: List[str]) -> List[Path]:
        uuid_dir = Config.IMAGES_DIR / uuid_str
        uuid_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, (data, ext) in enumerate(zip(images_bytes_list, exts_list), start=1):
            unique_suffix = uuidlib.uuid4().hex[:8]
            fname = f"{uuid_str}_{i}_{unique_suffix}{ext}"
            path = uuid_dir / fname
            path.write_bytes(data)
            saved_paths.append(path)
        return saved_paths

    @staticmethod
    def replace_images(uuid_str: str, images_bytes_list: List[bytes], exts_list: List[str]) -> List[Path]:
        uuid_dir = Config.IMAGES_DIR / uuid_str
        tmp_dir = Config.IMAGES_DIR / f".{uuid_str}_tmp"
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)

        try:
            for i, (data, ext) in enumerate(zip(images_bytes_list, exts_list), start=1):
                unique_suffix = uuidlib.uuid4().hex[:8]
                fname = f"{uuid_str}_{i}_{unique_suffix}{ext}"
                (tmp_dir / fname).write_bytes(data)

            # Atomic swap
            backup_dir = Config.IMAGES_DIR / f".{uuid_str}_backup"
            if uuid_dir.exists():
                if backup_dir.exists(): shutil.rmtree(backup_dir)
                uuid_dir.rename(backup_dir)
            
            tmp_dir.rename(uuid_dir)
            
            if backup_dir.exists(): shutil.rmtree(backup_dir)
            
            return StorageService.get_image_paths(uuid_str)
            
        except Exception:
            if tmp_dir.exists(): shutil.rmtree(tmp_dir)
            # Restore backup if needed
            possible_backup = Config.IMAGES_DIR / f".{uuid_str}_backup"
            if possible_backup.exists():
                if uuid_dir.exists(): shutil.rmtree(uuid_dir)
                possible_backup.rename(uuid_dir)
            raise

    @staticmethod
    def get_image_paths(uuid_str: str) -> List[Path]:
        uuid_dir = Config.IMAGES_DIR / uuid_str
        if not uuid_dir.exists(): return []
        files = [p for p in uuid_dir.iterdir() if p.is_file()]
        files.sort()
        return files
    
    @staticmethod
    def delete_images(uuid_str: str):
        uuid_dir = Config.IMAGES_DIR / uuid_str
        if uuid_dir.exists():
            shutil.rmtree(uuid_dir)


class DatabaseService:
    """Handles Reading/Writing NPZ files."""

    @staticmethod
    def get_db_path(email: str) -> Path:
        safe_email = email.replace("@", "_").replace(".", "_")
        return Config.BASE_DB_DIR / f"{safe_email}_db.npz"

    @staticmethod
    def get_classes_path(email: str) -> Path:
        safe_email = email.replace("@", "_").replace(".", "_")
        return Config.BASE_DB_DIR / f"{safe_email}_classes.json"

    @classmethod
    def load_db(cls, email: str) -> Tuple[List[Dict], np.ndarray]:
        path = cls.get_db_path(email)
        if not path.exists():
            return [], np.empty((0, 512), dtype=np.float32)
        
        try:
            data = np.load(path, allow_pickle=True)
            if "metadata" in data:
                meta_json = data["metadata"].tolist()
                if isinstance(meta_json, bytes): meta_json = meta_json.decode("utf-8")
                metadata = json.loads(meta_json)
                embeddings = data["embeddings"].astype(np.float32)
                return metadata, embeddings
            
            # Legacy Format Support
            if "labels" in data:
                return cls._migrate_legacy_db(data, path)
                
        except Exception:
            traceback.print_exc()
            
        return [], np.empty((0, 512), dtype=np.float32)

    @classmethod
    def _migrate_legacy_db(cls, data, path):
        labels = data["labels"].tolist()
        embeddings = data["embeddings"].astype(np.float32)
        metadata = []
        now = datetime.utcnow().isoformat() + "Z"
        for lbl in labels:
            if isinstance(lbl, bytes): lbl = lbl.decode("utf-8")
            roll, name = (lbl.split(":", 1) if ":" in lbl else (lbl, None))
            metadata.append({
                "id": str(uuid.uuid4()),
                "roll_no": roll,
                "name": name,
                "created_at": now,
                "updated_at": now
            })
        cls.save_db(path, metadata, embeddings)
        return metadata, embeddings

    @classmethod
    def save_db(cls, path: Path, metadata: List[Dict], embeddings: np.ndarray):
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        meta_json = json.dumps(metadata, ensure_ascii=False)
        np.savez_compressed(
            path, 
            metadata=np.array(meta_json), 
            embeddings=embeddings.astype(np.float32)
        )

    @classmethod
    def load_classes(cls, email: str) -> List[Dict]:
        path = cls.get_classes_path(email)
        if not path.exists(): return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @classmethod
    def save_classes(cls, email: str, classes: List[Dict]):
        path = cls.get_classes_path(email)
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(classes, f, ensure_ascii=False, indent=2)


class FaceService:
    def __init__(self):
        print("Initializing Face Analysis models...")
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(Config.DET_SIZE, Config.DET_SIZE))
        print("Models initialized.")

    def extract_faces(self, img: np.ndarray) -> List[Any]:
        return self.app.get(img)

    def extract_embedding(self, img: np.ndarray) -> Optional[np.ndarray]:
        faces = self.extract_faces(img)
        if not faces: return None
        # Get largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        return ImageUtils.l2_normalize(face.embedding)

    def annotate_image(self, img: np.ndarray, metadata: List[Dict], embeddings_db: np.ndarray) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
        """
        Returns: (recognized_students, annotated_full_img, annotated_unknown_only_img)
        """
        faces = self.extract_faces(img)
        recognized = []
        img_all = img.copy()
        img_unrec = img.copy()

        if embeddings_db.size == 0 or len(metadata) == 0:
            # Nothing to match against
            return [], img_all, img_unrec

        for face in faces:
            emb = ImageUtils.l2_normalize(face.embedding)
            sims = np.dot(embeddings_db, emb)
            idx = np.argmax(sims)
            score = float(sims[idx])
            
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box

            if score >= Config.MATCH_THRESHOLD:
                match = metadata[idx]
                name_disp = f"{match.get('name') or match.get('roll_no')} ({score:.2f})"
                recognized.append(match)
                
                # Draw Green
                cv2.rectangle(img_all, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_all, name_disp, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # Unknown
                name_disp = f"Unknown ({score:.2f})"
                # Draw Red on both
                cv2.rectangle(img_all, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_all, name_disp, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.rectangle(img_unrec, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img_unrec, name_disp, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return recognized, img_all, img_unrec


# =============================================================================
# SECTION 4: GLOBAL APP STATE (In-Memory Implementation)
# =============================================================================

class AppState:
    def __init__(self):
        self._db_lock = Lock()
        # Memory Cache: { "email": { "metadata": [], "embeddings": np.array, "classes": [] } }
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get_user_data(self, email: str) -> Dict[str, Any]:
        with self._db_lock:
            if email not in self._cache:
                meta, emb = DatabaseService.load_db(email)
                classes = DatabaseService.load_classes(email)
                self._cache[email] = {
                    "metadata": meta,
                    "embeddings": emb,
                    "classes": classes
                }
            return self._cache[email]

    def update_user_db(self, email: str, metadata: List[Dict], embeddings: np.ndarray):
        with self._db_lock:
            self._cache[email]["metadata"] = metadata
            self._cache[email]["embeddings"] = embeddings
            # Persist async or sync? For safety we do sync here to ensure durability
            DatabaseService.save_db(DatabaseService.get_db_path(email), metadata, embeddings)

    def update_user_classes(self, email: str, classes: List[Dict]):
        with self._db_lock:
            if email in self._cache:
                self._cache[email]["classes"] = classes
            else:
                # Should verify if initialized, but safe to just set
                self._cache[email] = self._cache.get(email, {})
                self._cache[email]["classes"] = classes
            DatabaseService.save_classes(email, classes)

    def invalidate(self, email: str):
         with self._db_lock:
             if email in self._cache:
                 del self._cache[email]

# Initialize Globals
face_service = FaceService()
app_state = AppState()

# =============================================================================
# SECTION 5: API APPLICATION & ROUTES
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    Config.setup()
    yield

app = FastAPI(lifespan=lifespan)

# --- Routes ---

@app.get("/")
def root():
    return {"message": "ProjectKAS API is running (Optimized Single-File)"}

@app.post("/signup")
def signup(email: str = Form(...)):
    try:
        path = DatabaseService.get_db_path(email)
        if not path.exists():
            DatabaseService.save_db(path, [], np.empty((0, 512), dtype=np.float32))
            return {"message": f"New DB created for {email}"}
        return {"message": f"DB already exists for {email}"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Signup failed: {str(e)}")

# Class Management

@app.get("/classes")
def list_classes(userEmail: str = Header(..., alias="User-Email")):
    classes = app_state.get_user_data(userEmail)["classes"]
    return {"total": len(classes), "classes": classes}

@app.post("/classes")
def create_class(name: str = Form(...), userEmail: str = Header(..., alias="User-Email")):
    name = name.strip()
    if not name: raise HTTPException(400, "Empty class name")
    
    data = app_state.get_user_data(userEmail)
    classes = data["classes"]
    
    if any(c["name"].lower() == name.lower() for c in classes):
        raise HTTPException(400, "Class exists")
        
    new_cls = {
        "id": f"cls_{uuid.uuid4().hex[:8]}",
        "name": name,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    classes.append(new_cls)
    app_state.update_user_classes(userEmail, classes)
    return new_cls

@app.put("/classes")
def update_class(classId: str = Form(...), name: str = Form(...), userEmail: str = Header(..., alias="User-Email")):
    name = name.strip()
    if not name: raise HTTPException(400, "Empty name")

    data = app_state.get_user_data(userEmail)
    classes = data["classes"]
    cls = next((c for c in classes if c["id"] == classId), None)
    if not cls: raise HTTPException(404, "Class not found")
    
    if any(c["name"].lower() == name.lower() and c["id"] != classId for c in classes):
        raise HTTPException(400, "Name collision")
        
    cls["name"] = name
    cls["updated_at"] = datetime.utcnow().isoformat() + "Z"
    app_state.update_user_classes(userEmail, classes)
    return cls

@app.delete("/classes")
def delete_class(classId: str = Form(...), userEmail: str = Header(..., alias="User-Email")):
    data = app_state.get_user_data(userEmail)
    classes = data["classes"]
    
    if not any(c["id"] == classId for c in classes):
        raise HTTPException(404, "Class not found")
        
    new_classes = [c for c in classes if c["id"] != classId]
    app_state.update_user_classes(userEmail, new_classes)
    
    # Cascade delete students?
    # Original logic: YES
    metadata = data["metadata"]
    embeddings = data["embeddings"]
    
    idxs_to_remove = [i for i, m in enumerate(metadata) if m.get("class") == classId]
    if idxs_to_remove:
        new_meta = [m for i, m in enumerate(metadata) if i not in idxs_to_remove]
        new_emb = np.delete(embeddings, idxs_to_remove, axis=0) if embeddings.size else embeddings
        app_state.update_user_db(userEmail, new_meta, new_emb)
        
    return {"message": "Class deleted"}

# Student Management

@app.post("/register")
async def register_student(
    Rollno: str = Form(...),
    studentName: str = Form(...),
    classId: str = Form(...),
    images: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    if not (Config.MIN_IMAGES <= len(images) <= Config.MAX_IMAGES):
        raise HTTPException(400, f"Upload {Config.MIN_IMAGES}-{Config.MAX_IMAGES} images")

    data = app_state.get_user_data(userEmail)
    classes = data["classes"]
    if not any(c["id"] == classId for c in classes):
         raise HTTPException(400, "Invalid classId")

    metadata = data["metadata"]
    embeddings_db = data["embeddings"]

    # Duplicate Check
    if any(m.get("class") == classId and m.get("roll_no") == Rollno  and m.get("name") == studentName for m in metadata):
        raise HTTPException(400, "Student already registered")

    # Process Images
    img_bytes_list = []
    ext_list = []
    valid_embeddings = []
    
    for file in images:
        raw, ext = await ImageUtils.get_bytes_and_ext(file)
        img_bytes_list.append(raw)
        ext_list.append(ext)
        img = ImageUtils.decode_image(raw)
        if img is None: continue
        
        emb = face_service.extract_embedding(img)
        if emb is not None:
            valid_embeddings.append(emb)

    if not valid_embeddings:
        raise HTTPException(400, "No faces detected in images")

    mean_emb = ImageUtils.l2_normalize(np.mean(np.stack(valid_embeddings), axis=0)).astype(np.float32) #type: ignore

    # Update Data
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
    
    # Atomic Update Logic
    new_meta = metadata + [new_record]
    new_emb = np.vstack([embeddings_db, mean_emb]) if embeddings_db.size else mean_emb[np.newaxis, :]
    
    try:
        # Save images first (filesystem)
        saved_paths = StorageService.save_images(student_id, img_bytes_list, ext_list)
        # Then update DB (In-memory + Disk)
        app_state.update_user_db(userEmail, new_meta, new_emb)
        
        return {
            "message": f"Registered {studentName}",
            "student": new_record,
            "saved_image_paths": [str(p) for p in saved_paths],
            "total_students": len(new_meta)
        }
    except Exception as e:
        traceback.print_exc()
        # Rollback images if they were saved but DB failed (unlikely with this order, but if Image save fails we just except)
        StorageService.delete_images(student_id)
        raise HTTPException(500, f"Registration failed: {str(e)}")

@app.get("/students")
def list_students(classId: str, userEmail: str = Header(...)):
    data = app_state.get_user_data(userEmail)
    students = [
        {"id": m["id"], "roll_no": m.get("roll_no"), "name": m.get("name")} 
        for m in data["metadata"] if m.get("class") == classId
    ]
    return {"message": f"Found {len(students)}", "students": students}

@app.delete("/studentDelete")
def delete_student(
    student_id: str = Form(...),
    classId: str = Form(...),
    userEmail: str = Header(...)
):
    data = app_state.get_user_data(userEmail)
    metadata = data["metadata"]
    embeddings = data["embeddings"]
    
    idx = next((i for i, m in enumerate(metadata) if m["id"] == student_id), None)
    if idx is None: raise HTTPException(404, "Student not found")
    
    if metadata[idx].get("class") != classId:
        raise HTTPException(403, "Class mismatch")
        
    meta_copy = list(metadata)
    meta_copy.pop(idx)
    
    emb_copy = embeddings
    if embeddings.size:
        emb_copy = np.delete(embeddings, idx, axis=0)
        
    app_state.update_user_db(userEmail, meta_copy, emb_copy)
    StorageService.delete_images(student_id)
    
    return {"message": "Deleted", "total": len(meta_copy)}

@app.put("/studentsUpdate")
def update_student(
    student_id: str = Form(...),
    classId: str = Form(...),
    roll_no: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
    userEmail: str = Header(...)
):
    data = app_state.get_user_data(userEmail)
    metadata = data["metadata"]
    
    idx = next((i for i, m in enumerate(metadata) if m["id"] == student_id), None)
    if idx is None: raise HTTPException(404, "Student not found")
    if metadata[idx].get("class") != classId: raise HTTPException(403, "Class mismatch")
    
    current = metadata[idx]
    new_roll = roll_no if roll_no is not None else current.get("roll_no")
    new_name = name if name is not None else current.get("name")
    
    # Collision check
    if any(m["id"] != student_id and m.get("class") == classId and m.get("roll_no") == new_roll and m.get("name") == new_name for m in metadata):
         raise HTTPException(400, "Update collision")

    # Clone and update
    meta_copy = list(metadata)
    meta_copy[idx] = dict(current)
    meta_copy[idx]["roll_no"] = new_roll
    meta_copy[idx]["name"] = new_name
    meta_copy[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    app_state.update_user_db(userEmail, meta_copy, data["embeddings"])
    return {"message": "Updated", "student": meta_copy[idx]}


@app.put("/students/re-enroll")
async def reenroll_student(
    student_id: str = Form(...),
    classId: str = Form(...),
    images: List[UploadFile] = File(...),
    userEmail: str = Header(...)
):
    if not (Config.MIN_IMAGES <= len(images) <= Config.MAX_IMAGES):
        raise HTTPException(400, "Image count invalid")

    data = app_state.get_user_data(userEmail)
    metadata = data["metadata"]
    embeddings = data["embeddings"]
    
    idx = next((i for i, m in enumerate(metadata) if m["id"] == student_id), None)
    if idx is None: raise HTTPException(404, "Student not found")
    if metadata[idx].get("class") != classId: raise HTTPException(403, "Class mismatch")

    # Process new images
    img_bytes_list = []
    ext_list = []
    valid_embs = []
    for file in images:
        raw, ext = await ImageUtils.get_bytes_and_ext(file)
        img_bytes_list.append(raw)
        ext_list.append(ext)
        img = ImageUtils.decode_image(raw)
        if img is not None:
            e = face_service.extract_embedding(img)
            if e is not None: valid_embs.append(e)

    if not valid_embs: raise HTTPException(400, "No faces found")
    
    mean_emb = ImageUtils.l2_normalize(np.mean(np.stack(valid_embs), axis=0)).astype(np.float32) #type: ignore
    
    # Update Embeddings
    emb_copy = embeddings.copy()
    emb_copy[idx] = mean_emb
    
    meta_copy = list(metadata)
    meta_copy[idx] = dict(meta_copy[idx])
    meta_copy[idx]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    # Save Images (Replace)
    try:
        saved_paths = StorageService.replace_images(student_id, img_bytes_list, ext_list)
        app_state.update_user_db(userEmail, meta_copy, emb_copy)
        return {"message": "Re-enrolled", "saved_image_paths": [str(p) for p in saved_paths]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Re-enroll failed: {str(e)}")

# Recognition

async def _process_recognition(files: List[UploadFile], userEmail: str, classId: str, debug: bool = False):
    data = app_state.get_user_data(userEmail)
    metadata = data["metadata"]
    embeddings_db = data["embeddings"]
    
    # Filter by class
    idxs = [i for i, m in enumerate(metadata) if m.get("class") == classId]
    if not idxs:
        # No students in this class
        filtered_meta = []
        filtered_emb = np.empty((0, 512), dtype=np.float32)
    else:
        filtered_meta = [metadata[i] for i in idxs]
        filtered_emb = embeddings_db[idxs]

    recognized_students = []
    unique_rec = {}
    
    # Debug images
    annotated_all = []
    annotated_unrec = []
    
    for file in files:
        try:
            raw, _ = await ImageUtils.get_bytes_and_ext(file)
            img = ImageUtils.decode_image(raw)
            if img is None: continue
            
            rec, img_a, img_u = face_service.annotate_image(img, filtered_meta, filtered_emb)
            
            for m in rec:
                key = m["id"]
                if key not in unique_rec:
                    unique_rec[key] = m
                    recognized_students.append(m)
            
            if debug:
                annotated_all.append(img_a)
                annotated_unrec.append(img_u)
                
        except Exception:
            traceback.print_exc()
            
    result = {"recognized_students": recognized_students}
    
    if debug and annotated_all:
        def merge(imgs):
            if not imgs: return None
            h = min(i.shape[0] for i in imgs)
            resized = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
            return cv2.hconcat(resized)
            
        merged_all = merge(annotated_all)
        merged_unrec = merge(annotated_unrec)
        
        result["annotated_all"] = ImageUtils.img_to_base64(merged_all)
        result["annotated_unrecognized"] = ImageUtils.img_to_base64(merged_unrec)
        
    return JSONResponse(result)

@app.post("/recognize")
async def recognize(files: List[UploadFile] = File(...), classId: str = Form(...), userEmail: str = Header(...)):
    return await _process_recognition(files, userEmail, classId, debug=False)

@app.post("/debugRecognize")
async def debug_recognize(files: List[UploadFile] = File(...), classId: str = Form(...), userEmail: str = Header(...)):
    return await _process_recognition(files, userEmail, classId, debug=True)

@app.post("/images/get")
def get_image(uuid: str = Form(...)):
    paths = StorageService.get_image_paths(uuid)
    if not paths: raise HTTPException(404, "No images")
    return Response(content=paths[0].read_bytes(), media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    # Run the application using uvicorn when executed directly
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

