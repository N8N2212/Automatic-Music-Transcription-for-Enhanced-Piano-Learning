# Automatic Piano Transcription: Converting Audio to Sheet Music for Enhanced Learning and Practice
Final Year project for year 2024/5
# Piano MP3 to Sheet Music

This repository contains a Python-based transcription system that converts beginner level piano MP3/WAV recordings into engraved sheet-music PDFs.

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
   git clone https://github.com/N8N2212/Automatic-Music-Transcription-for-Enhanced-Piano-Learning.git
   cd Automatic-Music-Transcription-for-Enhanced-Piano-Learning

## Usage

1. Run the Flask app:
   ```bash
   python App.py
2. In your browser, go to http://127.0.0.1:5000/ or the link provided in the terminal
3. Upload an MP3 file and wait for processing.
4. Click the download link to get your generated sheet-music PDF.
