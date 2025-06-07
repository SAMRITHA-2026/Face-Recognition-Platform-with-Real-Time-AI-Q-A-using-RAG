const express = require("express");
const axios = require("axios");
const http = require("http");
const cors = require("cors");
const { Server } = require("socket.io");

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: "*",
    },
});

// REST Proxy
app.post("/api/register_face", async (req, res) => {
    try {
        const response = await axios.post("http://localhost:5000/register_face", req.body);
        res.json(response.data);
    } catch (err) {
        res.status(500).json({ error: "Error in register_face" });
    }
});

app.post("/api/recognize_faces", async (req, res) => {
    try {
        const response = await axios.post("http://localhost:5000/recognize_faces", req.body);
        res.json(response.data);
    } catch (err) {
        res.status(500).json({ error: "Error in recognize_faces" });
    }
});

// WebSocket Proxy for RAG Query
io.on("connection", (socket) => {
    console.log("Client connected");

    socket.on("rag_query", async (msg) => {
        try {
            const response = await axios.post("http://localhost:5000/rag_query", { question: msg });
            socket.emit("rag_response", response.data.answer);
        } catch (err) {
            socket.emit("rag_response", "Error processing RAG query.");
        }
    });

    socket.on("disconnect", () => {
        console.log("Client disconnected");
    });
});

server.listen(4000, () => {
    console.log("Node.js server running on port 4000");
});
