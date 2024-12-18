#Import the libraries and read the audio file
import numpy as np
from scipy.io.wavfile import read as wavread
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import math

sr, data_read = wavread('dogs.wav')

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

def frequencies(audio):
    data_freq = np.fft.fftfreq(len(audio),0.7/sr)
    data_freq = data_freq[1:]
    return data_freq

def amplitude(audio):
    data_ampl = np.abs(np.fft.fft(audio))
    data_ampl = data_ampl[1:]
    return data_ampl
        
def energy(audio):
    data_amplitude = amplitude(audio)
    data_energy = data_amplitude ** 2
    return data_energy 

def energy_and_frequencies(data_freq, data_energy):
    energy_freq = {}
    for (i, freq) in enumerate(data_freq):
        if abs(freq) not in energy_freq:
            energy_freq[abs(freq)] = data_energy[i] * 2
    return energy_freq
    
def normalized_energy(audio):
    data_freq = frequencies(audio)
    data_energy = energy(audio)
    energy_freq = energy_and_frequencies(data_freq, data_energy)
    return energy_freq
    
def total_energy(energy_frequencies, start_band, end_band):
    total_energy = 0
    for f in energy_frequencies.keys():
        if start_band < f < end_band:
            total_energy += energy_frequencies[f]
    return total_energy

def median_filter (x, k):
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

def speech_detection(detected_windows):
    speech_window = 0.5 #half a second
    window = 0.02 #20 ms
    median_window=int(speech_window/window)
    if median_window%2==0: 
        median_window=median_window-1
    median_energy = median_filter(detected_windows[:,1], median_window)
    return median_energy

#Detects speech regions based on ratio between speech band energy and total energy.
#Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech)
def detect_speech(audio, sr):
    detected_windows = np.array([])
    window = 0.02 #20 ms
    overlap = 0.01 #10ms
    start_band = 300 #speech start band
    end_band = 3000 #speech end band
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
        energy_freq = normalized_energy(data_window)
        sum_voice_energy = total_energy(energy_freq, start_band, end_band)
        sum_full_energy = sum(energy_freq.values())
        speech_ratio = sum_voice_energy/sum_full_energy
        #Supposition: when there is a speech sequence we have ratio of energies more than the energy threshold
        speech_ratio = speech_ratio>energy_threshold
        detected_windows = np.append(detected_windows,[sample_start, speech_ratio])
        sample_start += overlap
    detected_windows = detected_windows.reshape(int(len(detected_windows)/2),2)
    detected_windows[:,1] = speech_detection(detected_windows)
    return detected_windows

def convert_to_labels(detected_windows):
    #Takes as input array of window numbers and speech flags from speech detection and convert speech flags to time intervals of speech.
    #Output is array of dictionaries with speech intervals.
    windows = []    
    speech_time = []
    is_speech = 0
    for window in detected_windows:
        if (window[1]==1.0 and is_speech==0): 
            is_speech = 1
            speech_label = {}
            speech_time_start = window[0] / sr
            speech_label['speech_begin'] = speech_time_start
            windows.append(window[0])
            print(window[0], speech_time_start)
                
        if (window[1]==0.0 and is_speech==1):
            is_speech = 0
            speech_time_end = window[0] / sr
            speech_label['speech_end'] = speech_time_end
            speech_time.append(speech_label)
            windows.append(window[0])
            print(window[0], speech_time_end)
    return speech_time, windows


raw_detection = detect_speech(audio, sr)

speech_labels, window_labels = convert_to_labels(raw_detection)
print(speech_labels)


for s in range(len(window_labels)):
    print(window_labels[s])


#Output: Saving splitted chunk into separate wave files
s = 0
output_prefix = "output_chunk"
while(s < len(window_labels)):
    start = window_labels[s]/sr
    end = window_labels[s+1]/sr
    print(start, end)
    chunk_audio = audio[int(window_labels[s]):int(window_labels[s+1])]
    output_file = f"{output_prefix}_{s}.wav"
    write(output_file, sr, chunk_audio)
    s = s+2


