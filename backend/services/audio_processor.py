import os
from gtts import gTTS
from typing import Optional

class AudioProcessor:
    """
    Handles clinical dictation and report speech synthesis (TTS).
    """
    
    def __init__(self, temp_dir: str = "data/audio_temp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

    def synthesize_report(self, text: str) -> Optional[str]:
        """
        Converts text report to an MP3 audio file.
        Returns the path to the generated MP3 file.
        """
        try:
            # Clean text from Markdown syntax for cleaner speech
            clean_text = text.replace("**", "").replace("#", "").replace("-", " ")
            
            tts = gTTS(text=clean_text, lang='en', slow=False)
            output_path = os.path.join(self.temp_dir, "consensus_report.mp3")
            
            # Remove old file if exists
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except Exception:
                    pass
                    
            tts.save(output_path)
            return output_path
        except Exception as e:
            print(f"Audio synthesis error: {e}")
            return None

    def transcribe_clinician_note(self, file_path: str) -> str:
        """
        Mock Whisper transcript logic. Extracts mutation tags from audio filenames 
        or transcribes clinical audio.
        """
        # In a real environment, this calls:
        # client.audio.transcriptions.create(model="whisper-1", file=file_handle)
        
        filename = os.path.basename(file_path).lower()
        if "egfr" in filename or "lung" in filename:
            return "Identify targeted therapy options for an EGFR L858R missense mutation."
        elif "braf" in filename or "melanoma" in filename:
            return "Run consensus analysis for a BRAF V600E alteration."
        elif "brca" in filename or "breast" in filename:
            return "Check clinical trials and PARP inhibitors for BRCA1 c.5266dupC."
        
        return "Retrieve clinical guidelines for a pathogenic BRCA1 mutation."
