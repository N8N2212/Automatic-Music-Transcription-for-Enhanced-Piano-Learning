# Automatic-Music-Transcription-for-Enhanced-Piano-Learning
Final Year project for year 2024/5
# Piano MP3 to Sheet Music

This repository contains a Python-based transcription system that converts monophonic and polyphonic piano MP3 recordings into engraved sheet-music PDFs.

## Features

- **Audio preprocessing** with FFmpeg → WAV conversion  
- **Onset detection** and **pitch estimation** using Librosa, SciPy, and Crepe (TensorFlow backend)  
- **Note and chord reconstruction** via Music21, exported to LilyPond for high-quality engraving  
- **Web interface** (Flask) for uploading MP3s and downloading PDFs  
- **Cross-platform** (Windows & Linux) with only CPU requirements  

## Prerequisites

- Python 3.12  
- FFmpeg (in PATH)  
- LilyPond (in PATH)  
- pip (for installing Python packages)  

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUserName/piano-mp3-to-sheet.git
   cd piano-mp3-to-sheet
