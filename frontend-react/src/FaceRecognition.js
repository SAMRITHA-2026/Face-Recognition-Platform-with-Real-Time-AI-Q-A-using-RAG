// src/components/FaceRegister.js

import React, { useState } from "react";

const FaceRegister = () => {
    const [name, setName] = useState("");
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (event) => {
        setFile(event.target.files[0]);
    };

    const handleRegister = async () => {
        if (!name || !file) {
            alert("Please enter a name and select an image.");
            return;
        }

        setLoading(true);

        const reader = new FileReader();
        reader.onloadend = async () => {
            const base64Image = reader.result.split(",")[1]; // Remove the data:image/... part
            try {
                await registerFace(name, base64Image);
            } catch (error) {
                console.error("Error in handleRegister:", error);
                alert("Failed to register face. Please try again.");
            } finally {
                setLoading(false);
            }
        };

        reader.onerror = () => {
            alert("Failed to read image file.");
            setLoading(false);
        };

        reader.readAsDataURL(file);
    };

    const registerFace = async (name, imageBase64) => {
        try {
            console.log("Sending POST to /api/register_face...");
            const response = await fetch("http://127.0.0.1:5000/api/register_face", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, image: imageBase64 }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log("Response from server:", data);
            alert(data.message);
        } catch (error) {
            console.error("Error registering face:", error);
            throw error; // rethrow so handleRegister can catch
        }
    };

    return (
        <div style={{
            border: "2px solid #ccc",
            padding: "20px",
            borderRadius: "12px",
            width: "400px",
            margin: "40px auto",
            textAlign: "center",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
            fontFamily: "Arial, sans-serif"
        }}>
            <h2 style={{ color: "#333", marginBottom: "20px" }}>🖼️ Register Face</h2>
            <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Enter Name"
                style={{
                    padding: "10px",
                    width: "80%",
                    marginBottom: "15px",
                    borderRadius: "6px",
                    border: "1px solid #aaa",
                    fontSize: "16px"
                }}
            />
            <br />
            <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ marginBottom: "10px" }}
            />
            <div style={{ fontSize: "14px", color: "#555", marginBottom: "10px" }}>
                {file ? `Selected file: ${file.name}` : "No file selected"}
            </div>
            <button
                onClick={handleRegister}
                disabled={loading}
                style={{
                    padding: "12px 24px",
                    cursor: loading ? "not-allowed" : "pointer",
                    backgroundColor: loading ? "#ccc" : "#4CAF50",
                    color: "#fff",
                    border: "none",
                    borderRadius: "6px",
                    fontSize: "16px",
                    transition: "background-color 0.3s"
                }}
            >
                {loading ? "Registering..." : "Register Face"}
            </button>
        </div>
    );
};

export default FaceRegister;
