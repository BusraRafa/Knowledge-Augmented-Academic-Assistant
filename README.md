# 🎓 Knowledge-Augmented Academic Assistant (AI Module)

This repository contains the **AI core** of a Knowledge-Augmented Academic Assistant designed to help students explore universities, get guidance, and receive accurate academic information.

The system is built using a **Retrieval-Augmented Generation (RAG)** approach, combining:
- 📊 Structured university data
- 🤖 Large Language Models (OpenAI)
- 🎙️ Voice input + transcription pipeline

---

## 🚀 Key Features

- 🤖 **AI Chatbot for Students**
  - Provides guidance on universities, programs, and decisions
  - Answers contextual and follow-up questions

- 📚 **RAG-Based Knowledge System**
  - Retrieves relevant university data
  - Augments LLM responses with real information
  - Improves factual accuracy and reduces hallucinations

- 🎙️ **Voice Input Support**
  - Accepts audio queries
  - Converts speech → text → AI response

- 🧠 **Context-Aware Conversations**
  - Maintains chat history
  - Generates more relevant responses over time

---

## 🧠 RAG Architecture (Core Idea)

This project follows a **Retrieval-Augmented Generation pipeline**, which works in two main stages:

### 1. Retrieval Stage
- User query is processed
- Relevant university data is selected from the internal dataset
- Context is prepared dynamically

### 2. Generation Stage
- Retrieved context + user query is sent to the LLM
- OpenAI model generates a grounded response

This approach improves reliability because:
- The model does not rely only on pre-trained knowledge
- It uses **real, up-to-date, domain-specific data**

> RAG systems enhance LLM outputs by injecting external knowledge before generation, improving accuracy and reducing hallucination. :contentReference[oaicite:0]{index=0}

---

## ⚙️ How the System Works

### Text Flow
1. User sends a query
2. System checks:
   - Chat history
   - University dataset
3. Relevant context is retrieved
4. OpenAI generates a structured response

### Voice Flow
1. User provides audio input (`.mp3`)
2. Audio is processed and transcribed
3. Transcribed text follows the same RAG pipeline
4. Final response is generated

---

## 🗂️ Project Structure

```
├── main.py                       # Core RAG chatbot pipeline (text)
├── modified_main.py              # Enhanced / structured response version
├── audio_main.py                 # Voice input + transcription + RAG pipeline
├── output/                       # Generated outputs and processed data
├── dummy1.mp3 - dummy5.mp3       # Sample audio files for testing
```

---


---

## 🧩 Tech Stack

- **Python**
- **OpenAI API (LLM + reasoning)**
- **Pydantic (structured outputs)**
- **Audio processing (local handling)**
- **RAG architecture (custom implementation)**

> Note: This project primarily uses **OpenAI** for intelligence. No external speech API dependency is strictly required in this version.

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/BusraRafa/Knowledge-Augmented-Academic-Assistant.git
cd Knowledge-Augmented-Academic-Assistant
```
### 2. Install dependencies
```
pip install -r requirements.txt
```
(If requirements.txt is missing, install manually: openai, python-dotenv, pydantic, pydub)

### 3. Set environment variables

Create a .env file:
```
OPENAI_API_KEY=your_openai_api_key
```
## ▶️ Usage
Run Text-Based Chatbot
```
python main.py
```
Run Voice-Based Assistant
```
python audio_main.py
```
## 🎧 Sample Audio

You can test the system using provided files:

- dummy1.mp3
- dummy2.mp3
- dummy3.mp3
- dummy4.mp3
- dummy5.mp3

## 📌 Notes
- This repository only contains the AI module
- Frontend, UI/UX, and full application integration are handled separately
- Designed to be integrated into:
  - Web apps
  - WhatsApp bots
  - Student platforms

## 💡 Future Improvements
- Better retrieval optimization (vector DB / embeddings)
- Multilingual support
- Better personalization for students
- Integration with live university APIs
- Text-to-Speech (voice output)
