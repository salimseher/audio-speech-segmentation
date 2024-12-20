# Audio Speech Segmentation
This repository contains a Python script for analyzing and processing audio files. The script detects speech regions in an audio file and splits it into smaller audio chunks based on detected speech intervals. It also provides useful insights into audio characteristics like amplitude, frequency, and energy.

## Features
1. Audio Analysis: Computes statistics such as frequency, amplitude, and energy of the input audio.
2. Speech Detection: Identifies speech regions in the audio using energy thresholds and band-limited energy analysis.
3. Audio Chunking: Splits the audio into chunks corresponding to detected speech intervals and saves them as separate .wav files.
4. Visualization: Plots the waveform of the input audio for quick inspection.

## Prerequisites
#### Before running the script, ensure you have the following installed:
Python 3.6+
#### Required libraries:
1. numpy
2. scipy
3. matplotlib

## Usage
1. Clone the repository.
2. Place your input audio file (e.g. input_audio.wav) in the same directory as the script.
3. Run the script:
> python ProgramFile.py
4. The script will:
  - Print audio statistics such as frequency and length.
  - Plot the waveform of the audio.
  - Detect speech intervals and save corresponding chunks as separate .wav files with filenames like output_chunk_0.wav.

## Functions Overview
#### 1. get_audio_frequency(audio)
  - Purpose: Identifies the frequency components present in the audio signal.
  - Explanation: This function uses the Fast Fourier Transform (FFT) to analyze the audio’s frequency domain. It reveals the "notes" or "tones" present in the sound.
#### 2. get_audio_amplitude(audio)
  - Purpose: Calculates the loudness of each frequency component.
  - Explanation: After the FFT, this function determines the magnitude (or strength) of each frequency. This shows which parts of the sound are louder.
#### 3. get_audio_energy(audio)
  - Purpose: Measures the overall strength of the audio signal.
  - Explanation: By squaring the amplitude values, this function emphasizes louder parts of the signal. This makes it easier to identify areas with speech or other significant sounds.
#### 4. speech_detector(detected_speech_windows)
  - Purpose: Smoothens detection results to reduce noise and improve accuracy.
  - Explanation: A median filter is applied to refine the detection, reducing sudden changes caused by noise or inconsistencies in the audio.
#### 5. speech_region_detection(audio, sr)
  - Purpose: Identifies where speech occurs in the audio file.
  - Explanation: The function divides the audio into overlapping windows (small chunks of time) and calculates the ratio of energy in the "speech band" (300 Hz to 3000 Hz) to total energy. If this ratio exceeds a threshold, the window is marked as speech.
#### 5. speech_labels(detected_speech_windows)
  - Purpose: Converts detected speech flags into readable time intervals.
  - Explanation: This function identifies transitions between "speech" and "no speech" in the detection results and outputs intervals of speech in terms of start and end times.

## Example Output
#### After running the script, you'll see:
1. A plot of the audio waveform.
2. Detected speech intervals printed as time ranges.
3. Chunked audio files saved in the directory as output_chunk_0.wav, output_chunk_1.wav, etc.

## Future Applications
#### The potential applications of this project include but not limited to:
1. Extracting speech segments for sentiment and tone analysis.
2. Assisting in analyzing and tracking speech progress.
3. Preparing datasets for machine learning models by segmenting and labeling speech.
4. Detecting speech in audio streams for monitoring purposes.

## License
This project is licensed under the MIT License.
