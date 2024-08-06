import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import json

def generate_pink_noise(signal_length, sampling_freq):
    # Generate white noise
    white_noise = np.random.randn(signal_length)
    # Fourier transform to get the frequency domain representation
    f_domain = np.fft.rfft(white_noise)
    # Generate the pink noise filter
    f = np.fft.rfftfreq(signal_length, d=1/sampling_freq)
    f[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(f)
    # Apply the pink filter to the white noise in the frequency domain
    f_domain = f_domain * pink_filter
    # Inverse Fourier transform to get back to the time domain
    pink_noise = np.fft.irfft(f_domain, n=signal_length)
    # Normalize the pink noise to be within [-1, 1]
    pink_noise = pink_noise / np.max(np.abs(pink_noise))
    return pink_noise


def AddFreqComp(signal, amp, freq, sampling_freq,noisez):
  signal_length = len(signal)
  time = (1/sampling_freq) * np.arange(0, signal_length, 1)
  noise = amp/2 * np.sin(2 * math.pi * freq * time + (2 * math.pi * np.random.uniform()))
  noisez = np.add(noisez,noise)
  noisy_signal = np.add(signal, noise)
  return noisy_signal,noisez


def poisson_process_event_locations(event_rate, data_time, sampling_freq):
    num_events = np.random.poisson(event_rate * data_time)
    time_between_events = np.random.exponential(1/event_rate, num_events)

    event_times = np.zeros(len(time_between_events))
    for i in range(len(time_between_events)):
        if i == 0:
            event_times[0] = time_between_events[0]
        else:
            if(np.sum(time_between_events[0:i+1]) < data_time):
              event_times[i] = np.sum(time_between_events[0:i+1])

    event_locs = np.round(sampling_freq * event_times)
    # Remove any zeros
    event_locs = event_locs[event_locs != 0]
    return event_locs


def bigaus_beads(bead_diameters, event_locs, data_time, num_points):
  """
  Args:
    
    event_locs: List of event locations as indices of total number of points
    data_time: Length in seconds of data stream being generated
    num_points: Total number of points in data stream (data_time * sampling_freq)

  Returns:
    events_clean: [1, num_points] array containing the synthetic data stream
  """
  
  params = {}
  for diameter in bead_diameters:
      
    params[diameter] = {}
      
    # Event Parameters
    params[diameter]['pos_amp_mean'] = -(diameter / 1.25e-4)**3
    params[diameter]['pos_amp_sd'] = abs(params[diameter]['pos_amp_mean']) * 0.40
    params[diameter]['neg_amp_mean'] = -(diameter / 9.3e-5)**3
    params[diameter]['neg_amp_sd'] = abs(params[diameter]['neg_amp_mean']) * 0.40
    params[diameter]['transit_time_mean'] = -293.4 * diameter + 0.004
    params[diameter]['transit_time_sd'] = params[diameter]['transit_time_mean'] * 0.157
      
  # Create time vector
  x = np.linspace(0, data_time, num_points)
  #save time
  # df = pd.DataFrame({'Time': x})
  # df.to_csv(f'./data/time_1.csv', index=False)
  # Initialize array
  events_clean = np.zeros(len(x))

  if type(event_locs) == int:
      num_events = 1
  else:
      num_events = len(event_locs)
    
  # Generate each event
  for i in range(num_events):

    diameter = bead_diameters[0]
      
    # Determine Bead Size
    # r = np.random.uniform(0, 1)
    # if r <= 0.5:
    #     diameter = bead_diameters[0]
    # else:
    #     diameter = bead_diameters[1]
    
    # Create event parameters
    a_1 = np.random.normal(params[diameter]['neg_amp_mean'], params[diameter]['neg_amp_sd']) # Negative Peak Amplitude
    a_2 = np.random.normal(params[diameter]['pos_amp_mean'], params[diameter]['pos_amp_sd']) # Positive Peak Amplitude
    d = np.random.normal(params[diameter]['transit_time_mean'], params[diameter]['transit_time_sd']) # Transit Time
    s = 0.3 * d # Peak Width

    if type(event_locs) == int:
        tc = event_locs
    else:
        tc = int(event_locs[i]) # Event center location

    # Generate event
    ex_1 = a_1 * np.exp(-(((x - x[tc]) + (d/2))**2)/(2*(s)**2))
    ex_2 = a_2 * np.exp(-(((x - x[tc]) - (d/2))**2)/(2*(s)**2))
    event = (ex_1 - ex_2)
    # Accumulate
    events_clean = events_clean + event
  return events_clean


def create_data_folder(folder_name, params):

    os.makedirs(f'{folder_name}')
    os.makedirs(f'{folder_name}/noisy')
    os.makedirs(f'{folder_name}/clean')
    
    with open(f'{folder_name}/parameters.json', 'w') as f:
        json.dump(params, f, indent=4)
     
     
def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples, samplerate):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

    
# ======================
# SIMULATION PARAMETERS
# ======================

# Dataset Parameters
num_streams = 1
sampling_freq = 7196
data_time = 30 # Time length of sample
num_points = round(sampling_freq * data_time) # length of each data stream

pink_noise_level = 5e-6  # I have found 5e-6 to be acceptable
white_noise_level = 5e-7

# Sets the factor by which the periodic noise amplitudes are multiplied by.
# In real data, different measuring frequencies have different noise levels:
# 2.2e-5 --> 3 MHz Measuring Freq
# 8.52e-5 --> 1 MHz Measuring Freq
# 1.35e-4 --> 500 kHz Measuring Freq
periodic_noise_level = 2.2e-5 
nl = '2_2e-5'

# Sets the mean bead diameter.
# 7 um -> 7.32e-6
# 4 um -> 4.19e-6
# 2 um -> 2.07e-6
# 500 nm -> 510e-9
# You can have up to two bead sizes in the list below
bead_diameters = [7.32e-6]

event_rate = 5 # Events per second

params = {
    'dataset_params': {
        'num_streams': num_streams,
        'sampling_freq': sampling_freq,
        'data_time': data_time,
        'pink_noise_level': pink_noise_level,
        'periodic_noise_level': periodic_noise_level
    },
    'event_params': {
        'event_rate': event_rate
    }
}

# Folder that data is saved to
folder = f'synthetic_data/7um/comparison/halfdense/noiselevel{nl}'
create_data_folder(folder, params)

data_streams_clean = np.empty([num_streams, num_points])

for k in range(num_streams):

  # UNCOMMENT FOR MULTI EVENT GENERATION   
  # Generate event locations according to a Poisson distribution
  event_locs = poisson_process_event_locations(event_rate, data_time, sampling_freq)
  
  # UNCOMMENT FOR SINGLE EVENT GENERATION
  # Create single event in center of stream
  #event_locs = round(num_points/2)

  # Generate Events
  events_clean = bigaus_beads(
      bead_diameters,
      event_locs,
      data_time,
      num_points
      )


  # Add data stream to array
  data_streams_clean[k, :] = events_clean
  df = pd.DataFrame({'Amplitude': events_clean})
  df.to_csv(f'{folder}/clean/clean_{k+1}.csv', index=False)
  
  print(f'{k+1} / {num_streams}')
  
print('Done!')

# ======================
# NOISING LOOP
# ======================

# Information extracted from FFT of actual signal.
# Amplitudes have been normalized by dividing each by the largest value

noise_info_amp = [
    0.0658684,
    0.2805874,
    0.4893336,
    0.0592756,
    0.1324801,
    0.1,
    0.06363636,
    0.06818182,
    1.,
    0.04045455,
    0.03772727,
    0.04363636,
    0.04272727,
    0.03454545,
    0.07727273,
    0.02590909,
    0.03636364,
    0.02454545,
    0.02090909,
    0.01636364,
    0.05909091
]
noise_info_amp = np.multiply(noise_info_amp, periodic_noise_level)

noise_info_freq = [
    3.5,
    6.6,
    10,
    20,
    26.7,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    160,
    170,
    180
]

# Time Vector
t = (1/sampling_freq)*np.arange(0, num_points, 1)

# Initialize noisy data array
data_streams_noisy = np.empty([num_streams, num_points])

print("Signal Noising\n-----------------------")

# Add noise to each data stream
for i in range(num_streams):

  signal = data_streams_clean[i, :]

  # Add periodic noise
  noisy_signal = signal
  noise = np.zeros(num_points)
  for j in range(len(noise_info_amp)):
    freq = noise_info_freq[j]
    amp = noise_info_amp[j]
    noisy_signal, noise = AddFreqComp(noisy_signal, amp, freq, sampling_freq, noise)

  # Add Pink Noise
  pink_noise = generate_pink_noise(num_points, sampling_freq)
  pink_noise = pink_noise * pink_noise_level
  noise = np.add(noise,pink_noise)
  noisy_signal = np.add(noisy_signal, pink_noise)
  
  # Add White noise
  # You can define the frequency range to generate the noise on (first two arguments below)
  #wn = band_limited_noise(0, 100, num_points, sampling_freq)
  #wn = wn / np.max(np.abs(wn))
  #wn *= white_noise_level
  #noisy_signal = np.add(noisy_signal, wn)
  
  # Add trend
  # noisy_signal, noise = AddFreqComp(noisy_signal, 1e-3, 0.0005, sampling_freq, noise)

  # save noise
  # df = pd.DataFrame({'Amplitude': noise})
  # df.to_csv(f'{folder}/noisy/{nl}_noisy_{i+1}.csv', index=False)

  # save noisy signal
  df = pd.DataFrame({'Amplitude': noisy_signal})
  df.to_csv(f'{folder}/noisy/noisy_{i+1}.csv', index=False)

  data_streams_noisy[i, :] = noisy_signal

  print(f'{i+1} / {num_streams}')

print('Done!')
