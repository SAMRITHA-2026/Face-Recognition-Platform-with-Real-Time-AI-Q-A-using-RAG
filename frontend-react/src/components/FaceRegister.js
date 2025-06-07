import React, { useState } from 'react';

const FaceRegister = () => {
    const [name, setName] = useState('');
    const [image, setImage] = useState(null);

    const handleRegister = async () => {
        if (!name || !image) {
            alert("Please enter name and select an image.");
            return;
        }

        const reader = new FileReader();
        reader.onloadend = async () => {
            const base64Image = reader.result.split(",")[1];

            const response = await fetch("http://127.0.0.1:5000/register_face", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, image: base64Image }),
            });

            const data = await response.json();
            alert(data.message);
        };

        reader.readAsDataURL(image);
    };

    return (
        <div className="card">
            <h2>📸 Register Face</h2>
            <input
                type="text"
                placeholder="Enter Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
            />
            <input
                type="file"
                accept="image/*"
                onChange={(e) => setImage(e.target.files[0])}
            />
            <button className="primary-btn" onClick={handleRegister}>
                Register Face
            </button>
        </div>
    );
};

export default FaceRegister;
