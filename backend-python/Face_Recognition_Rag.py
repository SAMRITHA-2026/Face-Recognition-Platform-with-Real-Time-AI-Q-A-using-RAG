# Face Recognition + Live Webcam + RAG QA

import streamlit as st
import face_recognition
import numpy as np
import os
import pickle
from datetime import datetime
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# Load environment variables
load_dotenv()
import google.generativeai as genai
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# App config
st.set_page_config("Face Recognition + QA", page_icon="🧑‍💻")
st.title("🧠 Face Recognition & Real-Time Q&A")

# Initialize session state for QA processing
if "processed" not in st.session_state:
    st.session_state.processed = False

# Initialize last recognized face state
if "last_recognized_face" not in st.session_state:
    st.session_state.last_recognized_face = "No face detected."

# Load or initialize face database
if not os.path.exists("face_db.pkl"):
    with open("face_db.pkl", "wb") as f:
        pickle.dump({}, f)

with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

# Helper function to save DB
def save_face_db():
    with open("face_db.pkl", "wb") as f:
        pickle.dump(face_db, f)

# Face Registration
st.sidebar.header("👤 Face Registration")
name_input = st.sidebar.text_input("Enter Name to Register Face")
register_image = st.sidebar.file_uploader("Upload Image for Registration", type=["png", "jpg", "jpeg"])
register_button = st.sidebar.button("📸 Register Face")

if register_button and name_input and register_image:
    try:
        image = face_recognition.load_image_file(register_image)
        face_locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, face_locations)

        if encodings:
            face_db[name_input] = {
                "encoding": encodings[0].tolist(),
                "timestamp": str(datetime.now())
            }
            save_face_db()
            st.sidebar.success(f"Face of {name_input} registered!")
        else:
            st.sidebar.error("No face detected in the uploaded image.")
    except Exception as e:
        st.sidebar.error(f"Error during registration: {str(e)}")

# Recognize faces in uploaded image
def recognize_faces_in_image(image_file):
    pil_image = Image.open(image_file)
    pil_image = pil_image.convert('RGB')
    image = np.array(pil_image).astype(np.uint8)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    encodings = face_recognition.face_encodings(rgb_image, face_locations)

    results = []
    for face_encoding in encodings:
        match_name = "Unknown"
        min_distance = 0.6

        for name, data in face_db.items():
            known_encoding = np.array(data["encoding"])
            distance = np.linalg.norm(known_encoding - face_encoding)
            if distance < min_distance:
                match_name = name
                min_distance = distance

        results.append(match_name)
    return results

# Text splitting for RAG
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Build FAISS Vector Store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Conversational Chain
def get_conversational_chain():
    prompt_template = """
    You are an assistant with access to the face registration database.
    Answer questions based on the following context.

    If the answer is not in the context, say:
    "I couldn't find this information in the provided data."

    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Generate Response
def generate_response(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        return response["output_text"]

    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "⚠️ Unable to generate answer. Possible quota exceeded or API error."

# Upload Image for Recognition
st.header("🖼️ Upload Image for Recognition")
uploaded_images = st.file_uploader(
    "Upload Images (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if st.button("Recognize Faces"):
    if not uploaded_images:
        st.warning("Please upload at least one image.")
    else:
        recognized_faces = []
        for image_file in uploaded_images:
            result = recognize_faces_in_image(image_file)
            recognized_faces.extend(result)

        if recognized_faces:
            st.success(f"Faces recognized: {', '.join(str(face) for face in recognized_faces)}")

            docs = []
            for name, data in face_db.items():
                text = f"Name: {name}\nRegistered on: {data['timestamp']}"
                docs.append(text)

            text_chunks = get_text_chunks("\n\n".join(docs))
            get_vector_store(text_chunks)
            st.session_state.processed = True
        else:
            st.warning("No known faces recognized.")

# Live Webcam Section

# Global frame capture state
FRAME_CAPTURE = {"frame": None}

# Video Processor
class FaceRecognitionProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_image)
        encodings = face_recognition.face_encodings(rgb_image, face_locations)

        detected_names = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
            match_name = "Unknown"
            min_distance = 0.6
            for name, data in face_db.items():
                known_encoding = np.array(data["encoding"])
                distance = np.linalg.norm(known_encoding - face_encoding)
                if distance < min_distance:
                    match_name = name
                    min_distance = distance

            detected_names.append(match_name)

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, match_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

        # Save current frame in correct BGR format
        FRAME_CAPTURE["frame"] = image.copy()

        # Save detected names to session state
        st.session_state.last_recognized_names = detected_names if detected_names else ["No face detected"]

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Live Webcam Recognition Tab
st.header("🎥 Live Face Recognition from Webcam")

# Initialize last_recognized_names state
if "last_recognized_names" not in st.session_state:
    st.session_state.last_recognized_names = ["No face detected"]

# Launch WebRTC with Processor
webrtc_streamer(
    key="face-recognition",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceRecognitionProcessor
)

# Process Last Captured Frame button
if st.button("📸 Process Last Captured Frame"):
    frame = FRAME_CAPTURE["frame"]

    if frame is not None:
        face_locations = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, face_locations)

        detected_names = []
        for face_encoding in encodings:
            match_name = "Unknown"
            min_distance = 0.6
            for name, data in face_db.items():
                known_encoding = np.array(data["encoding"])
                distance = np.linalg.norm(known_encoding - face_encoding)
                if distance < min_distance:
                    match_name = name
                    min_distance = distance

            detected_names.append(match_name)

        if detected_names:
            st.session_state.last_recognized_face = ", ".join(detected_names)
        else:
            st.session_state.last_recognized_face = "No face detected."
    else:
        st.session_state.last_recognized_face = "No frame captured yet."

# Display Detected Faces
st.subheader("Detected Faces:")
st.info(st.session_state.last_recognized_face)

# QA Section
if st.session_state.get("processed", False):
    st.header("💬 Ask Questions About Registered Faces")
    user_question = st.text_area(
        "Enter your question here",
        height=100,
        placeholder="Who was the last person registered?"
    )

    if st.button("Get Answer") and user_question:
        with st.spinner("Thinking..."):
            response = generate_response(user_question)
            st.subheader("Answer")
            st.write(response)

