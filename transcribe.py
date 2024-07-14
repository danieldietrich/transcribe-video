import whisper
import sys
import os

def print_usage_and_exit():
    print("Usage: python script_name.py <audio_file_path> <output_file_path>")
    sys.exit(1)

# Check if the command line argument is provided
if len(sys.argv) < 3:
    print_usage_and_exit()

# First command line argument is the input file stored in the variable audio_file_path
audio_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Verify if the provided path exists
if not os.path.isfile(audio_file_path):
    print(f"Error: The file '{audio_file_path}' does not exist.")
    print_usage_and_exit()

# Load the pre-trained Whisper model
model = whisper.load_model("base")

# Transcribe the audio file
result = model.transcribe(audio_file_path, fp16=False)

# Optionally, save the transcription to a text file
with open(output_file_path, "w") as file:
    file.write(result["text"])

print(f"Transcription saved to {output_file_path}")
