
#Import the libraries and read the audio file
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math

sr, data_read = wavread('input_audio.wav')

audio = data_read[:,0]

#print frequency
print("Frequency = ",sr,"Hz")

length = audio.shape[0] / sr
print(f"Length of the audio is = {length}s")

#Plot the audio file
time = np.linspace(0., length, audio.shape[0])
plt.plot(time, audio, label="Left channel")
plt.plot(time, audio, label="Right channel")
#plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [dB]")
plt.show()

#Some statistics to get started
max_amp = max(audio)
min_amp = min(audio)
length_amp = len(audio)
print(max_amp)
print(min_amp)
print(length_amp)

def get_audio_frequency(audio):
    data_frequency = np.fft.fftfreq(len(audio),0.7/sr)
    data_frequency = data_frequency[1:]
    return data_frequency

def get_audio_amplitude(audio):
    data_amplitude = np.abs(np.fft.fft(audio))
    data_amplitude = data_amplitude[1:]
    return data_amplitude
        
def get_audio_energy(audio):
    data_amplitude = get_audio_amplitude(audio)
    data_energy = data_amplitude ** 2
    return data_energy 

def get_energy_and_frequency(data_frequency, data_energy):
    energy_frequency = {}
    for (i, freq) in enumerate(data_frequency):
        if abs(freq) not in energy_frequency:
            energy_frequency[abs(freq)] = data_energy[i] * 2
    return energy_frequency
    
def get_normalized_energy(audio):
    data_frequency = get_audio_frequency(audio)
    data_energy = get_audio_energy(audio)
    energy_frequency = get_energy_and_frequency(data_frequency, data_energy)
    return energy_frequency
    
def audio_total_energy(energy_frequencies, speech_start_band, speech_end_band):
    total_energy = 0
    for ef in energy_frequencies.keys():
        if speech_start_band < ef < speech_end_band:
            total_energy += energy_frequencies[ef]
    return total_energy

def median_filter(x, z):
    assert z % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    z2 = (z - 1) // 2
    k = np.zeros ((len (x), z), dtype=x.dtype)
    k[:,z2] = x
    for i in range (z2):
        j = z2 - i
        k[j:,i] = x[:-j]
        k[:j,i] = x[0]
        k[:-j,-(i+1)] = x[j:]
        k[-j:,-(i+1)] = x[-1]
    return np.median(k, axis = 1)

def speech_detector(detected_speech_windows):
    speech_window = 0.5 #half a second
    window = 0.02 #20 ms
    median_window = int(speech_window/window)
    if median_window%2 == 0: 
        median_window = median_window - 1
    median_energy = median_filter(detected_speech_windows[:,1], median_window)
    return median_energy

#Detects speech regions based on ratio between speech band energy and total energy.
#Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech)
def speech_region_detection(audio, sr):
    detected_speech_windows = np.array([])
    window = 0.02 #20 ms
    overlap = 0.01 #10ms
    speech_start_band = 300 #speech start band
    speech_end_band = 3000 #speech end band
    energy_threshold = 0.35 #40% of energy in voice band
    window = int(sr * window)
    overlap = int(sr * overlap)
    data = audio
    sample_start = 0
    while (sample_start < (len(data) - window)):
        sample_end = sample_start + window
        if sample_end >= len(data): 
            sample_end = len(data)-1
        data_window = data[sample_start:sample_end]
        energy_frequency = get_normalized_energy(data_window)
        voice_energy = audio_total_energy(energy_frequency, speech_start_band, speech_end_band)
        full_energy = sum(energy_frequency.values())
        speech_ratio = voice_energy/full_energy
        #Supposition: the detected speech sequence has ratio of energies more than the energy threshold
        speech_ratio = speech_ratio > energy_threshold
        detected_speech_windows = np.append(detected_speech_windows,[sample_start, speech_ratio])
        sample_start += overlap
    detected_speech_windows = detected_speech_windows.reshape(int(len(detected_speech_windows)/2),2)
    detected_speech_windows[:,1] = speech_detector(detected_speech_windows)
    return detected_speech_windows

def speech_labels(detected_speech_windows):
    #Takes input as array of window numbers and speech flags from speech detection and converts speech flags to time intervals.
    #Output is array of dictionaries with speech intervals.
    speechwindows = []    
    time_speech = []
    is_speech = 0
    for window in detected_speech_windows:
        if (window[1]==1.0 and is_speech==0): 
            is_speech = 1
            speech_label = {}
            speech_time_start = window[0] / sr
            speech_label['speech_begin'] = speech_time_start
            speechwindows.append(window[0])
            print(window[0], speech_time_start)
                
        if (window[1]==0.0 and is_speech==1):
            is_speech = 0
            speech_time_end = window[0] / sr
            speech_label['speech_end'] = speech_time_end
            time_speech.append(speech_label)
            speechwindows.append(window[0])
            print(window[0], speech_time_end)
    return time_speech, speechwindows


speech_detection = speech_region_detection(audio, sr)

time_stamps, window_labels = speech_labels(speech_detection)
print(time_stamps)

#Output: Saving splitted chunk into separate wave files
s = 0
output_prefix = "output_chunk"
while(s < len(window_labels)):
    start = window_labels[s]/sr
    end = window_labels[s+1]/sr
    chunk_audio = audio[int(window_labels[s]):int(window_labels[s+1])]
    output_file = f"{output_prefix}_{s}.wav"
    write(output_file, sr, chunk_audio)
    s = s+2