from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from typing import List, Optional
import base64
import os
import uvicorn
import requests
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime
from pinecone import Pinecone
from openai import OpenAI
from pydub import AudioSegment
import speech_recognition as sr
from dotenv import load_dotenv
import json

load_dotenv()

# Define the response structure expected by the frontend
class ResponseModel(BaseModel):
    main_response: str
    follow_up_questions: List[str]
    
class AudioRequest(BaseModel):
    audioUrl: str  # Base64-encoded audio data
    config: Optional[dict] = {}
    savedFilePath: Optional[str] = ""
    
    @field_validator('audioUrl')
    def validate_audio_url(cls, v):
        if not v:
            raise ValueError('audioUrl cannot be empty')
        # Basic validation to ensure it looks like base64
        try:
            # Remove data URL prefix if present
            if ',' in v:
                v = v.split(',')[1]
            # Test decode a small portion
            base64.b64decode(v[:100])
            return v
        except Exception:
            raise ValueError('Invalid base64 audio data')

class TextRequest(BaseModel):
    text: str
    
    @field_validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('text cannot be empty')
        return v.strip()

# Initialize FastAPI app
app = FastAPI()

# Add custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request format. Please check your audio data encoding.",
            "errors": str(exc)
        }
    )

# Add custom exception handler for general exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    print(f"General error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to save uploaded audio
AUDIO_FILE_PATH = "temp_audio.wav"
AUDIO_SAVE_DIR = "audio_files"

# Create the directory if it doesn't exist
os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# Load API keys
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define Pinecone index name
INDEX_NAME = "asd-therapy-interactions"
index = pc.Index(INDEX_NAME)

def extract_features(audio_path, sr=22050):
    """
    Extract audio features using librosa.
    Returns a feature vector with MFCCs and other audio characteristics.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

        # Combine features into a single array
        features = np.hstack([mfccs, chroma, mel])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def speech_to_text(audio_url, config):
    """
    Convert speech to text using Google Cloud Speech-to-Text API.
    """
    try:
        # Create uploads directory if it doesn't exist
        uploads_dir = Path(__file__).parent / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        # Create audio recordings directory if it doesn't exist
        audio_dir = uploads_dir / "audio_recordings"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Generate a unique filename with timestamp and .wav extension
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"recording_{timestamp}.wav"
        file_path = audio_dir / file_name

        # Convert base64 string to binary and save to file
        # Handle data URL format (data:audio/wav;base64,...)
        if ',' in audio_url:
            audio_url = audio_url.split(',')[1]
            
        audio_data = base64.b64decode(audio_url)
        with open(file_path, 'wb') as audio_file:
            audio_file.write(audio_data)

        print(f"Audio file saved at: {file_path}")

        # Prepare the request to Google Speech-to-Text API
        api_url = "https://speech.googleapis.com/v1/speech:recognize"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-goog-api-key": os.getenv("GOOGLE_SPEECH_TO_TEXT_API_KEY")
        }
        payload = {
            "audio": {
                "content": audio_url
            },
            "config": config
        }

        # Make the request to Google Speech-to-Text API
        response = requests.post(api_url, json=payload, headers=headers)
        speech_results = response.json()
        print("Speech Results: ", speech_results)

        # Extract the transcribed text
        if "results" in speech_results and len(speech_results["results"]) > 0:
            transcribed_text = speech_results["results"][0]["alternatives"][0]["transcript"]
        else:
            transcribed_text = "No transcription available."

        return transcribed_text

    except Exception as err:
        print(f"Error converting speech to text: {err}")
        return f"Error: {str(err)}"

# Function to generate OpenAI embeddings
def generate_embedding(text: str) -> list:
    try:
        return client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def similarity_search(query_embedding: list, top_k: int = 5):
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results["matches"]
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []

# Function to combine all data for OpenAI query
def formulate_openai_query(text: str, context: list):
    try:
        # Format context from Pinecone results
        context_str = "\n".join([
            f"Child: {match['metadata']['text']}"
            for match in context if match.get('metadata', {}).get('role') == 'child'
        ])
        
        response_str = "\n".join([
            f"Assistant: {match['metadata']['text']}"
            for match in context if match.get('metadata', {}).get('role') == 'assistant'
        ])
        
        # Formulate query for OpenAI
        query = (
            f"The child said: \"{text}\".\n\n"
            f"Relevant past interactions:\n{context_str}\n\n"
            f"Assistant responses:\n{response_str}\n\n"
            f"Provide a response that is supportive and suitable for a child with ASD."
        )
        print("Generated OpenAI Query:")
        print(query)
        
        return query
    except Exception as e:
        print(f"Error formulating OpenAI query: {e}")
        return f"The child said: \"{text}\". Provide a supportive response for a child with ASD."

# Function to get response from OpenAI using chat models
def get_response_from_openai(prompt: str):
    try:
        system_instruction = (
            "You are a kind and supportive voice assistant designed to help a child with autism spectrum disorder (ASD). "
            "Your goal is to help the child understand their emotions, express their feelings, and improve their social interaction skills using calm and simple language. "
            "Avoid complex words, idioms, or abstract phrases. Use clear and gentle language that is easy for a child to understand. "
            "In your response, you may include one simple follow-up question only if it is directly relevant to what the child said. "
            "The follow-up question should be phrased as if the child is asking it themselves — from their own perspective. "
            "For example, if the child says 'I feel sad', your response might include a question like 'What can I do to feel better when I'm sad?'. "
            "Format your response using the following structure:\n\n"
            "Response: <Your main supportive response>\n"
            "Follow-up Question: <One simple and relevant question the child might ask next>"
        )


        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:spectrum-learner:AarQpVNF",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract the response text
        response_text = response.choices[0].message.content.strip()

        # Split the response into main response and follow-up questions
        if "Follow-up Question:" in response_text:
            main_response, follow_up_questions = response_text.split("Follow-up Question:")
            main_response = main_response.replace("Response:", "").strip()
            follow_up_questions = [follow_up_questions.strip()]
        else:
            # Handle case where there's no follow-up question
            main_response = response_text.replace("Response:", "").strip()
            follow_up_questions = []

        return {
            "main_response": main_response,
            "follow_up_questions": follow_up_questions
        }
    except Exception as e:
        print(f"Error getting OpenAI response: {e}")
        return {
            "main_response": "I'm sorry, I'm having trouble processing your request right now.",
            "follow_up_questions": ["Can you try asking me something else?"]
        }

# Alternative endpoint that handles raw request body
@app.post("/process-audio-raw")
async def process_audio_raw(request: Request):
    try:
        # Get raw body
        body = await request.body()
        
        # Try to parse as JSON
        try:
            data = json.loads(body.decode('utf-8'))
        except UnicodeDecodeError:
            # If body contains binary data, handle differently
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request format. Please send JSON data."}
            )
        
        # Extract audio URL from the parsed data
        audio_url = data.get('audioUrl', '')
        config = data.get('config', {})
        
        if not audio_url:
            raise HTTPException(status_code=400, detail="audioUrl is required")
        
        # Process the audio
        return await process_audio_logic(audio_url, config)
        
    except Exception as e:
        print(f"Error in process_audio_raw: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing audio: {str(e)}"}
        )

async def process_audio_logic(audio_url: str, config: dict):
    """Common logic for processing audio"""
    temp_file_path = None
    try:
        # Handle data URL format
        if ',' in audio_url:
            audio_url = audio_url.split(',')[1]
            
        # Decode base64 audio data
        audio_data = base64.b64decode(audio_url)

        # Create unique temporary file path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        temp_file_path = f"temp_audio_{timestamp}.wav"

        # Save the audio file
        with open(temp_file_path, "wb") as f:
            f.write(audio_data)
        print(f"Audio file saved to: {temp_file_path}")

        # Convert audio to WAV format using pydub
        try:
            audio = AudioSegment.from_file(temp_file_path)
            audio.export(temp_file_path, format="wav")
            print("Audio file converted to WAV format")
        except Exception as e:
            print(f"Warning: Could not convert audio format: {e}")

        # Convert speech to text using speech_recognition
        text = ""
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_file_path) as source:
                audio_data_sr = recognizer.record(source)
                text = recognizer.recognize_google(audio_data_sr)
                print(f"Recognized Text: {text}")
        except Exception as e:
            print(f"Error in speech recognition: {e}")
            # Fallback to Google Cloud Speech API if available
            if os.getenv("GOOGLE_SPEECH_TO_TEXT_API_KEY"):
                text = speech_to_text(audio_url, config)
            else:
                text = "Could not transcribe audio"
            
        # Get OpenAI response directly using the transcribed text
        system_instruction = (
            "You are a kind and supportive voice assistant designed to help a child with autism spectrum disorder (ASD). "
            "Your goal is to help the child understand their emotions, express their feelings, and improve their social interaction skills using calm and simple language. "
            "Avoid complex words, idioms, or abstract phrases. Use clear and gentle language that is easy for a child to understand. "
            "In your response, you may include one simple follow-up question only if it is directly relevant to what the child said. "
            "The follow-up question should be phrased as if the child is asking it themselves — from their own perspective. "
            "For example, if the child says 'I feel sad', your response might include a question like 'What can I do to feel better when I'm sad?'. "
            "Format your response using the following structure:\n\n"
            "Response: <Your main supportive response>\n"
            "Follow-up Question: <One simple and relevant question the child might ask next>"
        )


        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:spectrum-learner:AarQpVNF",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"The child said: \"{text}\". Provide a supportive response."}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract the response text
        response_text = response.choices[0].message.content.strip()

        # Split the response into main response and follow-up questions
        if "Follow-up Question:" in response_text:
            main_response, follow_up_questions = response_text.split("Follow-up Question:")
            main_response = main_response.replace("Response:", "").strip()
            follow_up_questions = [follow_up_questions.strip()]
        else:
            # Handle case where there's no follow-up question
            main_response = response_text.replace("Response:", "").strip()
            follow_up_questions = []

        return ResponseModel(
            main_response=main_response,
            follow_up_questions=follow_up_questions
        )

    except Exception as e:
        print(f"Error in processing audio: {e}")
        return ResponseModel(
            main_response=f"I'm sorry, I encountered an error while processing your audio. Please try again.",
            follow_up_questions=["Can you repeat what you said?"]
        )

    finally:
        # Clean up temporary audio file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                print(f"Error cleaning up file: {e}")

# Main endpoint to process audio file
@app.post("/process-audio", response_model=ResponseModel)
async def process_audio(request: AudioRequest):
    return await process_audio_logic(request.audioUrl, request.config)

@app.post("/process-text", response_model=ResponseModel)
async def process_text(request: TextRequest):
    try:
        # Get OpenAI response directly using the input text
        system_instruction = (
            "You are a kind and supportive voice assistant designed to help a child with autism spectrum disorder (ASD). "
            "Your goal is to help the child understand their emotions, express their feelings, and improve their social interaction skills using calm and simple language. "
            "Avoid complex words, idioms, or abstract phrases. Use clear and gentle language that is easy for a child to understand. "
            "In your response, you may include one simple follow-up question only if it is directly relevant to what the child said. "
            "The follow-up question should be phrased as if the child is asking it themselves — from their own perspective. "
            "For example, if the child says 'I feel sad', your response might include a question like 'What can I do to feel better when I'm sad?'. "
            "Format your response using the following structure:\n\n"
            "Response: <Your main supportive response>\n"
            "Follow-up Question: <One simple and relevant question the child might ask next>"
        )


        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal:spectrum-learner:AarQpVNF",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"The child said: \"{request.text}\". Provide a supportive response."}
            ],
            max_tokens=200,
            temperature=0.7
        )

        # Extract the response text
        response_text = response.choices[0].message.content.strip()

        # Split the response into main response and follow-up questions
        if "Follow-up Question:" in response_text:
            main_response, follow_up_questions = response_text.split("Follow-up Question:")
            main_response = main_response.replace("Response:", "").strip()
            follow_up_questions = [follow_up_questions.strip()]
        else:
            # Handle case where there's no follow-up question
            main_response = response_text.replace("Response:", "").strip()
            follow_up_questions = []

        return ResponseModel(
            main_response=main_response,
            follow_up_questions=follow_up_questions
        )
    except Exception as e:
        print(f"Error in processing text: {e}")
        return ResponseModel(
            main_response=f"I'm sorry, I encountered an error while processing your text. Please try again.",
            follow_up_questions=["Can you try asking me something else?"]
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Voice assistant is running"}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)