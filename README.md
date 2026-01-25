# Presence Cam – Backend
### FastAPI Monolithic Face Recognition Service

This repository contains the **backend service** for **Presence Cam**, an AI-powered smart attendance system. The backend is implemented as a **single-file FastAPI monolith**, intentionally designed for rapid development, low latency, and simple deployment, while still handling real-world concerns like rollback, data integrity, and multi-class isolation.

The service is responsible for **student registration, face recognition, class management, and image storage**, using **InsightFace** for deep learning–based facial embeddings.

---

## 🧠 Why Monolithic FastAPI?

This backend intentionally uses a **monolithic architecture** because:

- The system is inference-heavy rather than service-heavy
- FastAPI provides excellent async performance with minimal overhead
- Single deployment simplifies debugging and iteration
- Ideal for MVPs, academic projects, and early-stage products

The design avoids premature microservices and can be modularized later if scale requires it.

---

## 🏗 High-Level Architecture

Android App (Jetpack Compose)
|
| REST APIs
v
FastAPI Backend (Monolith)
|
|-- Image Decoding & Preprocessing
|-- Face Detection & Embedding (InsightFace)
|-- Similarity Matching
|
File-Based Persistence
├── user_dbs/ → Face embeddings + metadata (.npz)
├── stored_images/ → Student profile images
└── *_classes.json → Class metadata


---

## 🔧 Tech Stack

- **FastAPI** – API framework
- **InsightFace (buffalo_l)** – Face recognition model
- **OpenCV** – Image decoding and annotation
- **NumPy** – Embedding storage and vector math
- **Pillow + pillow-heif** – Image format compatibility (JPG, PNG, HEIC)
- **File-based persistence** (NPZ + JSON)

---

## 🧠 Face Recognition Pipeline

### Registration Flow
1. Student uploads **1–3 images**
2. Faces detected using InsightFace
3. Largest face selected per image
4. Face embeddings extracted and **L2-normalized**
5. Mean embedding computed across images
6. Embedding stored with student metadata

### Recognition Flow
1. Classroom image(s) uploaded
2. Faces detected
3. Embeddings generated
4. **Cosine similarity** used for matching
5. Threshold-based decision for identity resolution

**Embedding size:** 512  
**Similarity threshold:** 0.4 (configurable)

---

## 📁 Data Storage Design

### Per User
- **Face database:**  
  `user_dbs/<email>_db.npz`
- **Class metadata:**  
  `user_dbs/<email>_classes.json`

### Per Student
- **Stored images:**  
  `stored_images/<student_uuid>/`

### Why NPZ?
- Fast load times
- Simple versioning
- Easy rollback
- No external database dependency

---

## 🔐 Multi-Tenancy & Isolation

- Each user is identified via the `userEmail` HTTP header
- Databases and classes are **isolated per user**
- Students are **scoped by class**
- All write operations enforce class ownership checks

This prevents accidental cross-class or cross-user access.

---

## 🔌 API Endpoints

### User Setup
- `POST /signup`  
  Initializes a database for a new user

---

### Class Management
- `GET /classes`
- `POST /classes`
- `PUT /classes`
- `DELETE /classes`

Classes must be created before registering students.

---

### Student Management
- `POST /register`  
  Register a student with images
- `GET /students`  
  List students by class
- `PUT /studentsUpdate`  
  Update student metadata
- `DELETE /studentDelete`  
  Delete student and stored images
- `PUT /students/re-enroll`  
  Re-enroll student and replace images atomically

---

### Attendance Recognition
- `POST /recognize`  
  Recognize students from classroom images
- `POST /debugRecognize`  
  Same as recognize, but returns annotated images for debugging

---

### Image Utilities
- `POST /images/get`  
  Retrieve a stored student image by UUID

---

## 🔄 Data Integrity & Rollbacks

The backend is designed to **never leave corrupted state**:

- Image save failures trigger database rollback
- Re-enrollment replaces images atomically
- Old databases are backed up during replacement
- Backward compatibility supported for older DB formats

This is deliberate defensive engineering.

## Known Limitations
File-based storage (intentional for MVP)
No liveness detection (photo spoofing possible)
CPU-only inference
Single-instance deployment

##🔮 Future Improvements
Vector database for embeddings
Liveness detection
GPU inference support
Attendance analytics
Institution-level role management
Modularization if scale requires

---

## ⚙ Configuration

Key constants defined in `main.py`:

```python
DET_SIZE = 640
DET_SIZE_RECO = 640
MATCH_THRESHOLD = 0.4
MIN_IMAGES = 1
MAX_IMAGES = 3
BASE_DB_DIR = "user_dbs"
IMAGES_DIR = "stored_images" 


