import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY")

client = OpenAI(api_key=OPENAI_API_KEY)



def transcribe_and_save(file_path: str, save_directory: str = OUTPUT_DIRECTORY):
    """Transcribe an MP3 audio file using OpenAI's Whisper model and save the transcription."""
    try:
        
        with open(file_path, 'rb') as audio_file:
              
            translation = client.audio.translations.create(
                model="whisper-1",  
                file=audio_file,
            )
                   
        if translation.text:
            transcription = translation.text
          
            file_name = os.path.splitext(os.path.basename(file_path))[0] + ".txt"
            save_path = os.path.join(save_directory, file_name)

            
            os.makedirs(save_directory, exist_ok=True)

            
            with open(save_path, 'w') as file:
                file.write(transcription)

            print(f"Transcription saved at: {save_path}")
            return transcription
        else:
            return {"error": "No transcription result found"}

    except Exception as e:
        #return {"error": str(e)}
        return {"error": "An error occurred during transcription. Please try again later."}

if __name__ == "__main__":
    
    file_path = "./dummy5.mp3"  
    save_directory = './output'  

    result = transcribe_and_save(file_path, save_directory)
    print(result)