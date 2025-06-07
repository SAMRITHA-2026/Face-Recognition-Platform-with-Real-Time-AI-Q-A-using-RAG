# Face Recognition Platform with Real-Time AI Q&A using RAG 🎥🤖💬

A browser-based platform that enables users to:

✅ Register faces  
✅ Recognize faces in real-time from webcam stream (multi-face detection)  
✅ Ask natural language questions about registered faces (via RAG + LLM)

---

## Demo Screenshot 📸

![Screenshot 2025-06-07 235645](https://github.com/user-attachments/assets/a9b8a56d-3198-47f0-a484-13ff87b5d984)


*(Example: Elon Musk detected in live webcam feed, faces annotated.)*

---

## Features ✨

### 1️⃣ Face Registration Tab

- Upload face image via UI
- Face encoding extracted using `face_recognition` library
- Name and timestamp stored in `face_db.pkl` (can be extended to use any database)
- Multiple unique face registrations supported
![Untitled](https://github.com/user-attachments/assets/6ccb1a21-78c7-49b2-8c5c-94ffbc431e14)

### 2️⃣ Live Recognition Tab

- Streams webcam video using `streamlit-webrtc`
- Continuously scans each frame for known faces
- Overlays bounding boxes and names for each detected face
- Handles **multi-face detection** per frame
- Optimized for running 1–2 frames/sec to suit typical laptops
![Screenshot 2025-06-08 000531](https://github.com/user-attachments/assets/335ace56-2157-4b31-b8eb-e3aaafcda0b3)


### 3️⃣ Chat-Based Query Interface (RAG)

- Chat UI embedded in app
- User can ask questions like:
  - "Who was the last person registered?"
  - "At what time was Karthik registered?"
  - "How many people are currently registered?"
- Works via **FAISS + LangChain + LLM** (Gemini / OpenAI ChatGPT)
- Real-time RAG responses powered by vector similarity search + LLM generation

![image](https://github.com/user-attachments/assets/a3964596-c667-4f97-bb44-13f77ce33263)

![image](https://github.com/user-attachments/assets/02f35430-afdc-470d-a517-e807b8a06baf)


---

## Architecture 🏛️

```plaintext
/face-recognition-rag/
├── backend-python/          # Python (Flask or FastAPI) server
│   ├── app.py                # Face Registration, Recognition, RAG API
│   ├── face_db.pkl           # Face database (Pickle format or replace with DB)
│   ├── faiss_index/          # Vector index for RAG
│   ├── requirements.txt      # Python dependencies
│
├── backend-nodejs/           # Node.js WebSocket + API proxy server (optional)
│   ├── server.js              # Express.js + WebSocket
│   ├── package.json           # Node.js dependencies
│
├── frontend-react/           # React.js frontend app
│   ├── src/
│   ├── public/
│   ├── package.json
│
└── README.md                 # Project description and usage
