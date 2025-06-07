# Face Recognition Platform with Real-Time AI Q&A using RAG 🎥🤖💬

A browser-based platform that enables users to:

✅ Register faces  
✅ Recognize faces in real-time from webcam stream (multi-face detection)  
✅ Ask natural language questions about registered faces (via RAG + LLM)

---

## Demo Screenshot 📸

![Face Recognition Demo](./screenshot.png)

*(Example: Elon Musk detected in live webcam feed, faces annotated.)*

---

## Features ✨

### 1️⃣ Face Registration Tab

- Upload face image via UI
- Face encoding extracted using `face_recognition` library
- Name and timestamp stored in `face_db.pkl` (can be extended to use any database)
- Multiple unique face registrations supported

### 2️⃣ Live Recognition Tab

- Streams webcam video using `streamlit-webrtc`
- Continuously scans each frame for known faces
- Overlays bounding boxes and names for each detected face
- Handles **multi-face detection** per frame
- Optimized for running 1–2 frames/sec to suit typical laptops

### 3️⃣ Chat-Based Query Interface (RAG)

- Chat UI embedded in app
- User can ask questions like:
  - "Who was the last person registered?"
  - "At what time was Karthik registered?"
  - "How many people are currently registered?"
- Works via **FAISS + LangChain + LLM** (Gemini / OpenAI ChatGPT)
- Real-time RAG responses powered by vector similarity search + LLM generation

---

## Architecture 🏛️

```plaintext
Frontend (Streamlit)
   |
WebRTC Video Stream ↔ FaceRecognitionProcessor (Python + OpenCV + face_recognition)
   |
Face Registration ↔ Face DB (Pickle or any DB)
   |
Chat Interface ↔ RAG Engine (LangChain + FAISS + LLM)
