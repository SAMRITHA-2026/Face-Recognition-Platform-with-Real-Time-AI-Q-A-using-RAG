import React, { useState } from 'react';

const RagChat = () => {
    const [question, setQuestion] = useState('');
    const [answer, setAnswer] = useState('');

    const handleAsk = async () => {
        if (!question) {
            alert("Please enter a question.");
            return;
        }

        const response = await fetch("http://127.0.0.1:5000/rag_query", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });

        const data = await response.json();
        setAnswer(data.answer || "No answer.");
    };

    return (
        <div className="card">
            <h2>💬 RAG Chat</h2>
            <textarea
                placeholder="Enter your question here..."
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
            ></textarea>
            <button className="primary-btn" onClick={handleAsk}>
                Ask
            </button>

            {answer && (
                <div className="results">
                    <h4>Answer:</h4>
                    <p>{answer}</p>
                </div>
            )}
        </div>
    );
};

export default RagChat;
