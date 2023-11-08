import zombie_imp
import csv
import serial
import time 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import pywt
import numpy as np
from scipy import stats
import statistics
from scipy.interpolate import interp1d
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import find_peaks
from scipy.fft import fft
from scipy.fftpack import dct
from scipy.signal import welch

class RingBuffer:
    def __init__(self, size):
        self.size = size
        self.data = [None] * size
        self.head = 0
        self.tail = 0
        self.count = 0

    def append(self, item):
        if self.count < self.size:
            self.data[self.tail] = item
            self.tail = (self.tail + 1) % self.size
            self.count += 1
        else:
            # If the buffer is full, overwrite the oldest data
            self.data[self.head] = item
            self.head = (self.head + 1) % self.size
            self.tail = (self.tail + 1) % self.size

    def get(self):
        return [self.data[(self.head + i) % self.size] for i in range(self.count)]
    
    def clear(self):
        self.data = [None] * self.size
        self.head = 0
        self.tail = 0
        self.count = 0

    def __len__(self):
        return self.count


# read the csv file
df_1 = pd.read_csv('final_dataset/2s_interval/flat.csv')
df_2 = pd.read_csv('final_dataset/2s_interval/vertical.csv')
df_3 = pd.read_csv('final_dataset/2s_interval/up.csv')
df_4 = pd.read_csv('final_dataset/2s_interval/down.csv')
df_5 = pd.read_csv('final_dataset/2s_interval/up_down.csv')
df_6 = pd.read_csv('final_dataset/2s_interval/down_up.csv')
df_7 = pd.read_csv('final_dataset/2s_interval/flat_inverse.csv')
df_8 = pd.read_csv('final_dataset/2s_interval/vertical_inverse.csv')

# get the data of each solar cell
solar_cell_1_flat = df_1['Cell 1']
solar_cell_2_flat = df_1['Cell 2']

solar_cell_1_vertical = df_2['Cell 1']
solar_cell_2_vertical = df_2['Cell 2']

solar_cell_1_up = df_3['Cell 1']
solar_cell_2_up = df_3['Cell 2']

solar_cell_1_down = df_4['Cell 1']
solar_cell_2_down = df_4['Cell 2']

solar_cell_1_up_down = df_5['Cell 1']
solar_cell_2_up_down = df_5['Cell 2']

solar_cell_1_down_up = df_6['Cell 1']
solar_cell_2_down_up = df_6['Cell 2']

solar_cell_1_flat_inverse = df_7['Cell 1']
solar_cell_2_flat_inverse = df_7['Cell 2']

solar_cell_1_vertical_inverse = df_8['Cell 1']
solar_cell_2_vertical_inverse = df_8['Cell 2']

# apply DWT for the current value of each solar cell
coeffs_1_flat = pywt.wavedec(solar_cell_1_flat, 'haar', level=5) #coeffs_1[0] contains the first-level approximation coefficient of solar_cell_1, coeffs_1[1] contains the first-level detail coefficient of solar_cell_1
coeffs_2_flat = pywt.wavedec(solar_cell_2_flat, 'haar', level=5)

coeffs_1_vertical = pywt.wavedec(solar_cell_1_vertical, 'haar', level=5)
coeffs_2_vertical = pywt.wavedec(solar_cell_2_vertical, 'haar', level=5)

coeffs_1_up = pywt.wavedec(solar_cell_1_up, 'haar', level=5)
coeffs_2_up = pywt.wavedec(solar_cell_2_up, 'haar', level=5)

coeffs_1_down = pywt.wavedec(solar_cell_1_down, 'haar', level=5)
coeffs_2_down = pywt.wavedec(solar_cell_2_down, 'haar', level=5)

coeffs_1_up_down = pywt.wavedec(solar_cell_1_up_down, 'haar', level=5)
coeffs_2_up_down = pywt.wavedec(solar_cell_1_up_down, 'haar', level=5)

coeffs_1_down_up = pywt.wavedec(solar_cell_1_down_up, 'haar', level=5)
coeffs_2_down_up = pywt.wavedec(solar_cell_1_down_up, 'haar', level=5)

coeffs_1_flat_inverse = pywt.wavedec(solar_cell_1_flat_inverse, 'haar', level=5)
coeffs_2_flat_inverse = pywt.wavedec(solar_cell_1_flat_inverse, 'haar', level=5)

coeffs_1_vertical_inverse = pywt.wavedec(solar_cell_1_vertical_inverse, 'haar', level=5)
coeffs_2_vertical_inverse = pywt.wavedec(solar_cell_1_vertical_inverse, 'haar', level=5)

# should be changed based on reality
threshold = 0.5

# For cell 1
approx_coeff_1_flat = coeffs_1_flat[0]
detail_coeff_1_flat = coeffs_1_flat[1]
approx_coeff_1_vertical = coeffs_1_vertical[0]
detail_coeff_1_vertical = coeffs_1_vertical[1]

## Apply a threshold on the detail coefficient on signals of cell 1
detail_coeff_thresh_1_flat = pywt.threshold(detail_coeff_1_flat, threshold, mode='soft')
detail_coeff_thresh_1_vertical = pywt.threshold(detail_coeff_1_vertical, threshold, mode='soft')

## Reconstruct the signal for different gestures of cell 1
coeffs_denoised_1_flat = [approx_coeff_1_flat, detail_coeff_thresh_1_flat]
solar_cell_denoised_1_flat = pywt.waverec(coeffs_denoised_1_flat, 'haar')
coeffs_denoised_1_vertical = [approx_coeff_1_vertical, detail_coeff_thresh_1_vertical]
solar_cell_denoised_1_vertical = pywt.waverec(coeffs_denoised_1_vertical, 'haar')

# For cell 2
approx_coeff_2_flat = coeffs_2_flat[0]
detail_coeff_2_flat = coeffs_2_flat[1]
approx_coeff_2_vertical = coeffs_2_vertical[0]
detail_coeff_2_vertical = coeffs_2_vertical[1]

## Apply a threshold on the detail coefficient of signals of cell 2
detail_coeff_thresh_2_flat = pywt.threshold(detail_coeff_2_flat, threshold, mode='soft')
detail_coeff_thresh_2_vertical = pywt.threshold(detail_coeff_2_vertical, threshold, mode='soft')

## Reconstruct the signal for different gestures of cell 2
coeffs_denoised_2_flat = [approx_coeff_2_flat, detail_coeff_thresh_2_flat]
solar_cell_denoised_2_flat = pywt.waverec(coeffs_denoised_2_flat, 'haar')
coeffs_denoised_2_vertical = [approx_coeff_2_vertical, detail_coeff_thresh_2_vertical]
solar_cell_denoised_2_vertical = pywt.waverec(coeffs_denoised_2_vertical, 'haar')


# plot the comparison of single before and after denoising
plt.figure(figsize=(12, 8))

plt.subplot(8, 2, 1)
#plt.plot(solar_cell_1_flat[17000: 25500])
plt.title('Cell 1 - Right to Left Flat Hand')

plt.subplot(8, 2, 2)
#plt.plot(solar_cell_2_flat[17000: 25500])
plt.title('Cell 2 - Right to Left Flat Hand')

plt.subplot(8, 2, 3)
#plt.plot(solar_cell_1_vertical)
plt.title('Cell 1 - Right to Left Vertical Hand')

plt.subplot(8, 2, 4)
#plt.plot(solar_cell_2_vertical)
plt.title('Cell 2 - Right to Left Vertical Habd')

plt.subplot(8, 2, 5)
#plt.plot(solar_cell_1_up)
plt.title('Cell 1 - Up Hand)')

plt.subplot(8, 2, 6)
#plt.plot(solar_cell_2_up)
plt.title('Cell 2 - Up Hand')


plt.subplot(8, 2, 7)
#plt.plot(solar_cell_1_down)
plt.title('Cell 1 - Down Hand')

plt.subplot(8, 2, 8)
#plt.plot(solar_cell_2_down)
plt.title('Cell 2 - Down Hand')

plt.subplot(8, 2, 9)
#plt.plot(solar_cell_1_up_down)
plt.title('Cell 1 - Up Down Hand')

plt.subplot(8, 2, 10)
#plt.plot(solar_cell_2_up_down)
plt.title('Cell 2 - Up Down Hand')

plt.subplot(8, 2, 11)
#plt.plot(solar_cell_1_down_up)
plt.title('Cell 1 - Down Up Hand')

plt.subplot(8, 2, 12)
#plt.plot(solar_cell_2_down_up)
plt.title('Cell 2 - Down Up Habd')

plt.subplot(8, 2, 13)
#plt.plot(solar_cell_1_flat_inverse)
plt.title('Cell 1 - Left to Right Flat Hand)')

plt.subplot(8, 2, 14)
#plt.plot(solar_cell_2_flat_inverse)
plt.title('Cell 2 - Left to Right Flat Hand')


plt.subplot(8, 2, 15)
#plt.plot(solar_cell_1_vertical_inverse)
plt.title('Cell 1 - Left to Right Vertical Hand')

plt.subplot(8, 2, 16)
#plt.plot(solar_cell_2_vertical_inverse)
plt.title('Cell 2 - Left to Right Vertical Hand')


plt.tight_layout()
#plt.show()

##Segmenting
#
#

class CellProcessor:
    def __init__(self):
        self.last5s = RingBuffer(1500)
        self.historyBuffer = RingBuffer(100)
        self.sliceArray = []
        self.busyReading = False
        self.avgMean = 0
        self.historyMean = 100000
        self.samplingStep = 0
        self.nrOfSlices = 0
        self.slice_arr = [] # all the segmentation slices is stored in this list for one signal

    def process_signal(self, signal):
        for sigVal in signal:
            if (not self.busyReading):
                self.last5s.append(sigVal)
                self.avgMean = np.mean(self.last5s.get())
                self.historyBuffer.append(sigVal)
                self.historyMean = np.mean(self.historyBuffer.get())

            if self.historyMean < 0.85 * self.avgMean and self.samplingStep < 1700:
                self.busyReading = True
                if not self.sliceArray:
                    self.sliceArray = self.historyBuffer.get()
                else:
                    self.sliceArray.append(sigVal)
                self.samplingStep += 1
            elif self.busyReading:
#                 #plt.plot(self.sliceArray)
#                 #plt.show()
                self.slice_arr.append(self.sliceArray)
                print("Finished 1 slice")
                self.samplingStep = 0
                self.busyReading = False
                self.historyMean = 1000000
                self.historyBuffer.clear()
                self.sliceArray = []
                self.nrOfSlices += 1

# Create the instance
processor_1_flat = CellProcessor()
processor_1_vertical = CellProcessor()
processor_1_up = CellProcessor()
processor_1_down = CellProcessor()
# processor_1_up_down = CellProcessor()
# processor_1_down_up = CellProcessor()
processor_1_flat_inverse = CellProcessor()
processor_1_vertical_inverse = CellProcessor()

processor_2_flat = CellProcessor()
processor_2_vertical = CellProcessor()
processor_2_up = CellProcessor()
processor_2_down = CellProcessor()
# processor_2_up_down = CellProcessor()
# processor_2_down_up = CellProcessor()
processor_2_flat_inverse = CellProcessor()
processor_2_vertical_inverse = CellProcessor()

# process the signal
processor_1_flat.process_signal(solar_cell_1_flat)
processor_1_vertical.process_signal(solar_cell_1_vertical)
processor_1_up.process_signal(solar_cell_1_up)
processor_1_down.process_signal(solar_cell_1_down)
# processor_1_up_down.process_signal(solar_cell_1_up_down)
# processor_1_down_up.process_signal(solar_cell_1_down_up)
processor_1_flat_inverse.process_signal(solar_cell_1_flat_inverse)
processor_1_vertical_inverse.process_signal(solar_cell_1_vertical_inverse)

processor_2_flat.process_signal(solar_cell_2_flat)
processor_2_vertical.process_signal(solar_cell_2_vertical)
processor_2_up.process_signal(solar_cell_2_up)
processor_2_down.process_signal(solar_cell_2_down) 
# processor_2_up_down.process_signal(solar_cell_2_up_down)
# processor_2_down_up.process_signal(solar_cell_2_down_up)
processor_2_flat_inverse.process_signal(solar_cell_2_flat_inverse)
processor_2_vertical_inverse.process_signal(solar_cell_2_vertical_inverse)


#
# STARTING PROCESSING SIGNAL for features
#
#
def interpolate_signal(signal, target_length=512):
    original_time = np.linspace(0, 1, len(signal))
    interpolator = interp1d(original_time, signal, kind='linear')
    upscaled_time = np.linspace(0, 1, target_length)
    return interpolator(upscaled_time)

def denoise_slice(slice):
    # should be changed based on reality
    threshold = 0.4
    # denoise the slice first, and then interpolation, and then calculate the detail coefficients. use these coefficients to calculate min, max mean
    coeffs = pywt.wavedec(slice, 'haar', level=5)
    approx_coeffs = coeffs[0]
    detail_coeffs = coeffs[1]
    
    # Apply a threshold on the detail coefficient on signals of cell 1
    detail_coeff_thresh = pywt.threshold(detail_coeffs, threshold, mode='soft')
    
    # Reconstruct the signal for different gestures of cell 1
    coeffs_denoised = [approx_coeffs, detail_coeff_thresh]
    slice_denoised = pywt.waverec(coeffs_denoised, 'haar')
    return slice_denoised

# denoise the slice first, and then interpolation, and then calculate the detail coefficients. use these coefficients to calculate min, max mean
def extract_features_from_slice(slice):    
    coeffs = pywt.wavedec(slice, 'haar', level=5)
    std_dev = statistics.stdev(coeffs[1])
    mean_val = np.mean(coeffs[1])
    min_val = np.min(coeffs[1])
    max_val = np.max(coeffs[1])
    difference = max_val - min_val
    return std_dev, mean_val, min_val, max_val, difference

# denoise the slice firstly, and then calculate statistical values directly. Z-score first gives a better performance!!!!!!!!!
def extract_features_from_slice_denoised(slice):
    # Extract the features
    slice = stats.zscore(slice)
    std_dev = statistics.stdev(slice)
    mean_val = np.mean(slice)
    min_val = np.min(slice)
    max_val = np.max(slice)
    difference = max_val - min_val
    # DOMINANT FREQUENCY
#     coeffs = pywt.wavedec(slice, 'haar', level=2)
    coeffs = [[0],slice]
    f, Pxx = welch(coeffs[1], fs=1.0, nperseg=1024)
    dominant_frequency = f[np.argmax(Pxx)]
    # Calculate the spectral centroid
    spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    return std_dev, mean_val, min_val, max_val, difference, dominant_frequency, spectral_centroid

def extract_features_remove_positives(slice):
    #Remove positives
    slice[slice > 0.0] = 0
    std_dev = statistics.stdev(slice)
    mean_val = np.mean(slice)
    min_val = np.min(slice)
    max_val = np.max(slice)
    difference = max_val - min_val
    # Define the window size
    window_size = 10

    max_energy = 0
    max_position = 0

    for i in range(len(slice) - window_size + 1):
        # Calculate energy in the current window
        window_energy = np.sum(slice[i:i + window_size] ** 2)

        # Check if this window has higher energy
        if window_energy > max_energy:
            max_energy = window_energy
            max_position = i

    print(f"Position with most energy: {max_position}")
    return std_dev, mean_val, min_val, max_val, difference, max_position
    
def process_solar_cell(sliceArray):
    features = {
        # the features below are calculated with DWT coefficients
        'std_dev': [],
        'mean_val': [],
        'min_val': [],
        'max_val': [],
        'difference': [],
        'dom_freq': [],
        'spectral_centroid': [],
        # the features below are calculated with denoised slices
        'std_dev_signal': [],
        'mean_val_signal': [],
        'min_val_signal': [],
        'max_val_signal': [],
        'difference_signal': [],
        'NP_std_dev': [],
        'NP_mean_val': [],
        'NP_min_val': [],
        'NP_max_val': [],
        'NP_difference': [],
        'max_position': [],
#         'leftRight': []
    }

    for slice in sliceArray:
        denoised_slice = denoise_slice(slice)
        interpolated_slice = interpolate_signal(denoised_slice)
        interpolated_slice_zscore = stats.zscore(interpolated_slice)

        std_dev, mean_val, min_val, max_val, difference = extract_features_from_slice(interpolated_slice)
        NP_std_dev, NP_mean_val, NP_min_val, NP_max_val, NP_difference, max_position = extract_features_remove_positives(interpolated_slice_zscore)
        std_dev_signal, mean_val_signal, min_val_signal, max_val_signal, difference_signal, dom_freq, spectral_centroid = extract_features_from_slice_denoised(denoised_slice)

        features['std_dev'].append(std_dev)
        features['mean_val'].append(mean_val)
        features['min_val'].append(min_val)
        features['max_val'].append(max_val)
        features['difference'].append(difference)
        features['dom_freq'].append(dom_freq)
        features['spectral_centroid'].append(spectral_centroid)


        features['std_dev_signal'].append(std_dev_signal)
        features['mean_val_signal'].append(mean_val_signal)
        features['min_val_signal'].append(min_val_signal)
        features['max_val_signal'].append(max_val_signal)
        features['difference_signal'].append(difference_signal)
        
        features['NP_std_dev'].append(NP_std_dev)
        features['NP_mean_val'].append(NP_mean_val)
        features['NP_min_val'].append(NP_min_val)
        features['NP_max_val'].append(NP_max_val)
        features['NP_difference'].append(NP_difference)
        features['max_position'].append(max_position)

       
        count = 0
        iterationSize = 2
        if (count < iterationSize):
#             coeffs = pywt.wavedec(interpolated_slice, 'db2', level=5)
            coeffs = [[0], interpolated_slice_zscore]
            #plt.plot(coeffs[1])
            #plt.show()
#             histPlot(coeffs[1])
#             powerPlot(coeffs[1])
            count+=1
        elif (count == iterationSize):
            count += 1
    print("done with set")

    return features

# Process of cell 1
features_solar_1_flat = process_solar_cell(processor_1_flat.slice_arr)
features_solar_1_vertical = process_solar_cell(processor_1_vertical.slice_arr)
features_solar_1_up = process_solar_cell(processor_1_up.slice_arr)
features_solar_1_down = process_solar_cell(processor_1_down.slice_arr)
# features_solar_1_up_down = process_solar_cell(processor_1_up_down.slice_arr)
# features_solar_1_down_up = process_solar_cell(processor_1_down_up.slice_arr)
features_solar_1_flat_inverse = process_solar_cell(processor_1_flat_inverse.slice_arr)
features_solar_1_vertical_inverse = process_solar_cell(processor_1_vertical_inverse.slice_arr)

# Process of cell 2
features_solar_2_flat = process_solar_cell(processor_2_flat.slice_arr)
features_solar_2_vertical = process_solar_cell(processor_2_vertical.slice_arr)
features_solar_2_up = process_solar_cell(processor_2_up.slice_arr)
features_solar_2_down = process_solar_cell(processor_2_down.slice_arr)
# features_solar_2_up_down = process_solar_cell(processor_2_up_down.slice_arr)
# features_solar_2_down_up = process_solar_cell(processor_2_down_up.slice_arr)
features_solar_2_flat_inverse = process_solar_cell(processor_2_flat_inverse.slice_arr)
features_solar_2_vertical_inverse = process_solar_cell(processor_2_vertical_inverse.slice_arr)

# Output
# print("Flat Hand cell 1")
# print(features_solar_1_flat)
# print("Vertical Hand cell 1")
# print(features_solar_1_vertical)
# print("Flat Hand cell 2")
# print(features_solar_2_flat)
# print("Vertical Hand cell 2")
# print(features_solar_2_vertical)



################################## KNN TESTER
df_t1 = pd.read_csv('final_dataset/test/flat.csv')
solar_cell_1_flat_test = df_t1['Cell 1']
solar_cell_2_flat_test = df_t1['Cell 2']

df_t2 = pd.read_csv('final_dataset/test/vertical.csv')
solar_cell_1_vertical_test = df_t2['Cell 1']
solar_cell_2_vertical_test = df_t2['Cell 2']

df_t3 = pd.read_csv('final_dataset/test/flat_inverse.csv')
solar_cell_1_flat_inverse_test = df_t3['Cell 1']
solar_cell_2_flat_inverse_test = df_t3['Cell 2']

df_t4 = pd.read_csv('final_dataset/test/vertical_inverse.csv')
solar_cell_1_vertical_inverse_test = df_t4['Cell 1']
solar_cell_2_vertical_inverse_test = df_t4['Cell 2']
# PREPARE THE TEST DATA
processor_1_flat_test = CellProcessor()
processor_1_vertical_test = CellProcessor()
# processor_1_up = CellProcessor()
# processor_1_down = CellProcessor()
# processor_1_up_down = CellProcessor()
# processor_1_down_up = CellProcessor()
processor_1_flat_inverse_test = CellProcessor()
processor_1_vertical_inverse_test = CellProcessor()

# Segment them
processor_1_flat_test.process_signal(solar_cell_1_flat_test)
processor_1_vertical_test.process_signal(solar_cell_1_vertical_test)
# processor_1_up.process_signal(solar_cell_1_up)
# processor_1_down.process_signal(solar_cell_1_down)
# processor_1_up_down.process_signal(solar_cell_1_up_down)
# processor_1_down_up.process_signal(solar_cell_1_down_up)
processor_1_flat_inverse_test.process_signal(solar_cell_1_flat_inverse_test)
processor_1_vertical_inverse_test.process_signal(solar_cell_1_vertical_inverse_test)

# Extract features
features_solar_1_flat_test = process_solar_cell(processor_1_flat_test.slice_arr)
features_solar_1_vertical_test = process_solar_cell(processor_1_vertical_test.slice_arr)
# features_solar_1_up = process_solar_cell(processor_1_up.slice_arr)
# features_solar_1_down = process_solar_cell(processor_1_down.slice_arr)
# features_solar_1_up_down = process_solar_cell(processor_1_up_down.slice_arr)
# features_solar_1_down_up = process_solar_cell(processor_1_down_up.slice_arr)
features_solar_1_flat_inverse_test = process_solar_cell(processor_1_flat_inverse_test.slice_arr)
features_solar_1_vertical_inverse_test = process_solar_cell(processor_1_vertical_inverse_test.slice_arr)

def createTestSet(ListOfData):
    #Create test set
    Y_test_1 = []
    # Use the training data
    Y_test_1 += ["Flat"] * len(features_solar_1_flat_test[ListOfData[0]])
    Y_test_1 += ["Vertical"] * len(features_solar_1_vertical_test[ListOfData[0]])
    # X_train_up_1 = np.column_stack((features_solar_1_up[x_data],features_solar_1_up[y_data], features_solar_1_up[z_data]))
    # X_train_down_1 = np.column_stack((features_solar_1_down[x_data], features_solar_1_down[y_data], features_solar_1_down[z_data]))
    # X_train_up_down_1 = np.column_stack((features_solar_1_up_down[x_data],features_solar_1_up_down[y_data], features_solar_1_up_down[z_data]))
    # X_train_down_up_1 = np.column_stack((features_solar_1_down_up[x_data], features_solar_1_down_up[y_data], features_solar_1_down_up[z_data]))

    Y_test_1 += ["Flat Inverse"] * len(features_solar_1_flat_inverse_test[ListOfData[0]])
    Y_test_1 +=  ["Vertical Inverse"] * len(features_solar_1_vertical_inverse_test[ListOfData[0]])
    # Create empty columns for columnStack
    X_test_flat_inverse_1 = np.array([features_solar_1_flat_inverse_test[key] for key in ListOfData]).T
    X_test_vertical_1 = np.array([features_solar_1_vertical_test[key] for key in ListOfData]).T
    X_test_flat_1 = np.array([features_solar_1_flat_test[key] for key in ListOfData]).T
    X_test_vertical_inverse_1 = np.array([features_solar_1_vertical_inverse_test[key] for key in ListOfData]).T
    
    X_test_1 = np.concatenate((X_test_flat_1, X_test_vertical_1, X_test_flat_inverse_1, X_test_vertical_inverse_1), axis=0)
    
    return X_test_1, Y_test_1

def testAllFeaturesKNN(ListOfData):
#         # The features used for training model
#         x_data = 'min_val_signal'
#         y_data = 'max_val_signal'
#         z_data = 'mean_val_signal'

        Y_train_1 = []
        # Use the training data
        Y_train_1 += ["Flat"] * len(features_solar_1_flat[ListOfData[0]])
#         X_train_flat_1 = np.column_stack((features_solar_1_flat[x_data],features_solar_1_flat[y_data], features_solar_1_flat[z_data], features_solar_1_flat[a_data]))
#         print(X_train_flat_1)
        Y_train_1 += ["Vertical"] * len(features_solar_1_vertical[ListOfData[0]])
#         X_train_vertical_1 = np.column_stack((features_solar_1_vertical[x_data], features_solar_1_vertical[y_data], features_solar_1_vertical[z_data], features_solar_1_vertical[a_data]))
        # X_train_up_1 = np.column_stack((features_solar_1_up[x_data],features_solar_1_up[y_data], features_solar_1_up[z_data]))
        # X_train_down_1 = np.column_stack((features_solar_1_down[x_data], features_solar_1_down[y_data], features_solar_1_down[z_data]))
        # X_train_up_down_1 = np.column_stack((features_solar_1_up_down[x_data],features_solar_1_up_down[y_data], features_solar_1_up_down[z_data]))
        # X_train_down_up_1 = np.column_stack((features_solar_1_down_up[x_data], features_solar_1_down_up[y_data], features_solar_1_down_up[z_data]))

        Y_train_1 += ["Flat Inverse"] * len(features_solar_1_flat_inverse[ListOfData[0]])
#         X_train_flat_inverse_1 = np.column_stack((features_solar_1_flat_inverse[x_data],features_solar_1_flat_inverse[y_data], features_solar_1_flat_inverse[z_data], features_solar_1_flat_inverse[a_data]))
        Y_train_1 +=  ["Vertical Inverse"] * len(features_solar_1_vertical_inverse[ListOfData[0]])
    
        # Create empty columns for columnStack
        X_train_flat_1 = np.array([features_solar_1_flat[key] for key in ListOfData]).T
        X_train_vertical_1 = np.array([features_solar_1_vertical[key] for key in ListOfData]).T
        X_train_flat_inverse_1 = np.array([features_solar_1_flat_inverse[key] for key in ListOfData]).T
        X_train_vertical_inverse_1 = np.array([features_solar_1_vertical_inverse[key] for key in ListOfData]).T

        
        X_train_1 = np.concatenate((X_train_flat_1, X_train_vertical_1, X_train_flat_inverse_1, X_train_vertical_inverse_1), axis=0)
#         print(X_train_1)
        # Create test set:
        X_test_1, Y_test_1 = createTestSet(ListOfData)
        # Create new KNN model with 3 features
        knn_model_1 = KNeighborsClassifier(n_neighbors=5)  
        knn_model_1.fit(X_train_1, Y_train_1)  # train the model

        y_pred = knn_model_1.predict(X_test_1)

        # Evaluate model performance using accuracy (you can use other metrics)

        accuracy = accuracy_score(Y_test_1, y_pred)
#         print(f"Accuracy using: {accuracy:.2f} -> {ListOfData}")
        return accuracy

from sklearn.metrics import accuracy_score
# REMOVE WARNING
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

listOfKeys = list(features_solar_1_flat.keys())

# x_data = 'min_val_signal'
# y_data = 'max_val_signal'
# z_data = 'mean_val_signal'
# testAllFeaturesKNN(x_data, y_data, z_data)

allFeaturesPerformanceDF = pd.DataFrame(columns=['accuracy', 'x_data', 'y_data', 'z_data', 'a_data', 'b_data'])

for key1 in listOfKeys:
        for key2 in listOfKeys:
            for key3 in listOfKeys:
                for key4 in listOfKeys:
                    for key5 in listOfKeys:
                        if (key1 == (key2 or key3 or key4 or key5) or 
                            key2 == (key1 or key3 or key4 or key5) or 
                            key3 == (key1 or key2 or key4 or key5) or
                            key4 == (key1 or key2 or key3 or key5) or
                            key5 == (key1 or key2 or key3 or key4)):
                                continue

                        x_data = key1
                        y_data = key2
                        z_data = key3
                        a_data = key4
                        b_data = key5
                        ListOfData = [key1, key2, key3, key4, key5]

                        accuracy = testAllFeaturesKNN(ListOfData)
                        new_row = {
                            'accuracy': accuracy,
                            'x_data': x_data,
                            'y_data': y_data,
                            'z_data': z_data,
                            'a_data': a_data,
                            'b_data': b_data
                        }

                        # Append the new row to the DataFrame
                        allFeaturesPerformanceDF = allFeaturesPerformanceDF.append(new_row, ignore_index=True)
                    
# Sort by highest accuracy
allFeaturesPerformanceDF = allFeaturesPerformanceDF.sort_values(by='accuracy', ascending=False)
# Reset the index to maintain a sequential index
allFeaturesPerformanceDF = allFeaturesPerformanceDF.reset_index(drop=True)
pd.set_option('display.max_rows', None)  # Show all rows
print(allFeaturesPerformanceDF.head(100))
# # Scatter plot for all gestures with different colors
# fig_1 = plt.figure(figsize=(10, 8))
# ax_1 = fig_1.add_subplot(projection='3d')

# # Create a color map for the gestures
# colors = {'flat_right_left': 'red', 'vertical_right_left': 'blue', 'up': 'green', 'down': 'purple', 'up&down': 'orange', 'down&up': 'brown', 'flat_left_right': 'pink', 'vertical_left_right': 'gray'}

# # Feature Extraction for all gestures
# min_1, std_1, mean_1 = X_train_1[:, 0], X_train_1[:, 1], X_train_1[:, 2]

# # Plot the figure for all gestures with different colors
# for label in colors:
#     indices = [i for i, x in enumerate(Y_train_1) if x == label]
#     ax_1.scatter(min_1[indices], std_1[indices], mean_1[indices], c=colors[label], label=label)

# # Add axises labels
# ax_1.set_xlabel('Min')
# ax_1.set_ylabel('Max')
# ax_1.set_zlabel('Mean')

# # Add legend
# ax_1.legend()

# #plt.show()


# # Train the KNN model for cell 2
# X_train_flat_2 = np.column_stack((features_solar_2_flat[x_data],features_solar_2_flat[y_data], features_solar_2_flat[z_data]))
# X_train_vertical_2 = np.column_stack((features_solar_2_vertical[x_data], features_solar_2_vertical[y_data], features_solar_2_vertical[z_data]))
# # X_train_up_2 = np.column_stack((features_solar_2_up[x_data],features_solar_2_up[y_data], features_solar_2_up[z_data]))
# # X_train_down_2 = np.column_stack((features_solar_2_down[x_data], features_solar_2_down[y_data], features_solar_2_down[z_data]))
# # X_train_up_down_2 = np.column_stack((features_solar_2_up_down[x_data],features_solar_2_up_down[y_data], features_solar_2_up_down[z_data]))
# # X_train_down_up_2 = np.column_stack((features_solar_2_down_up[x_data], features_solar_2_down_up[y_data], features_solar_2_down_up[z_data]))
# X_train_flat_inverse_2 = np.column_stack((features_solar_2_flat_inverse[x_data],features_solar_2_flat_inverse[y_data], features_solar_2_flat_inverse[z_data]))
# X_train_vertical_inverse_2 = np.column_stack((features_solar_2_vertical_inverse[x_data], features_solar_2_vertical_inverse[y_data], features_solar_2_vertical_inverse[z_data]))
# # X_train_flat_2 = np.column_stack((features_solar_2_flat['min_val'],features_solar_2_flat['max_val'], features_solar_2_flat['mean_val']))
# # X_train_vertical_2 = np.column_stack((features_solar_2_vertical['min_val'], features_solar_2_vertical['max_val'], features_solar_2_vertical['mean_val']))
# X_train_2 = np.concatenate((X_train_flat_2, X_train_vertical_2, X_train_up_2, X_train_down_2, X_train_up_down_2, X_train_down_up_2, X_train_flat_inverse_2, X_train_vertical_inverse_2), axis=0)

# Y_train_2 = ['flat_right_left'] * 40 + ['vertical_right_left']*40 + ['up']*40 + ['down'] * 40 + ['up&down']*40 + ['down&up']*40 + ['flat_left_right']*40 + ['vertical_left_right']*40
# knn_model_2 = KNeighborsClassifier(n_neighbors=3)  
# knn_model_2.fit(X_train_2, Y_train_2)  # train the model

# fig_2 = plt.figure(figsize=(10, 8))
# ax_2 = fig_2.add_subplot(projection='3d')

# # Feature Extraction of cell 2
# min_2_flat, std_2_flat, mean_2_flat = X_train_flat_2[:, 0], X_train_flat_2[:, 1], X_train_flat_2[:, 2]
# min_2_vertical, std_2_vertical, mean_2_vertical = X_train_vertical_2[:, 0], X_train_vertical_2[:, 1], X_train_vertical_2[:, 2]
# # min_2_up, std_2_up, mean_2_up = X_train_up_2[:, 0], X_train_up_2[:, 1], X_train_up_2[:, 2]
# # min_2_down, std_2_down, mean_2_down = X_train_down_2[:, 0], X_train_down_2[:, 1], X_train_down_2[:, 2]
# # min_2_up_down, std_2_up_down, mean_2_up_down = X_train_up_down_2[:, 0], X_train_up_down_2[:, 1], X_train_up_down_2[:, 2]
# # min_2_down_up, std_2_down_up, mean_2_down_up = X_train_down_up_2[:, 0], X_train_down_up_2[:, 1], X_train_down_up_2[:, 2]
# min_2_flat, std_2_flat, mean_2_flat = X_train_flat_inverse_2[:, 0], X_train_flat_inverse_2[:, 1], X_train_flat_inverse_2[:, 2]
# min_2_vertical, std_2_vertical, mean_2_vertical = X_train_vertical_inverse_2[:, 0], X_train_vertical_inverse_2[:, 1], X_train_vertical_inverse_2[:, 2]

# # Plot the figure for cell 2
# ax_2.scatter(min_2_flat, std_2_flat, mean_2_flat, c='r', label='Flat')
# ax_2.scatter(min_2_vertical, std_2_vertical, mean_2_vertical, c='b', label='Vertical')
# # ax_2.scatter(min_2_up, std_2_up, mean_2_up, c='green', label='Up')
# # ax_2.scatter(min_2_down, std_2_down, mean_2_down, c='purple', label='Down')
# # ax_2.scatter(min_2_up_down, std_2_up_down, mean_2_up_down, c='orange', label='Up & Down')
# # ax_2.scatter(min_2_down_up, std_2_down_up, mean_2_down_up, c='brown', label='Down & Up')
# ax_2.scatter(min_2_flat, std_2_flat, mean_2_flat, c='pink', label='Flat Inverse')
# ax_2.scatter(min_2_vertical, std_2_vertical, mean_2_vertical, c='gray', label='Vertical Inverse')

# # Add axises labels
# ax_2.set_xlabel('Min')
# ax_2.set_ylabel('Max')
# ax_2.set_zlabel('Mean')

# ax_2.legend()

# #plt.show()


from sklearn.metrics import accuracy_score
# REMOVE WARNING
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

listOfKeys = list(features_solar_1_flat.keys())

# x_data = 'min_val_signal'
# y_data = 'max_val_signal'
# z_data = 'mean_val_signal'
# testAllFeaturesKNN(x_data, y_data, z_data)

allFeaturesPerformanceDF = pd.DataFrame(columns=['accuracy', 'x_data', 'y_data', 'z_data', 'a_data', 'b_data'])

for key1 in listOfKeys:
        for key2 in listOfKeys:
            for key3 in listOfKeys:
                for key4 in listOfKeys:
                    for key5 in listOfKeys:
                        if (key1 == (key2 or key3 or key4 or key5) or 
                            key2 == (key1 or key3 or key4 or key5) or 
                            key3 == (key1 or key2 or key4 or key5) or
                            key4 == (key1 or key2 or key3 or key5) or
                            key5 == (key1 or key2 or key3 or key4)):
                                continue

                        x_data = key1
                        y_data = key2
                        z_data = key3
                        a_data = key4
                        b_data = key5
                        ListOfData = [key1, key2, key3, key4, key5]

                        accuracy = testAllFeaturesKNN(ListOfData)
                        new_row = {
                            'accuracy': accuracy,
                            'x_data': x_data,
                            'y_data': y_data,
                            'z_data': z_data,
                            'a_data': a_data,
                            'b_data': b_data
                        }

                        # Append the new row to the DataFrame
                        allFeaturesPerformanceDF = allFeaturesPerformanceDF.append(new_row, ignore_index=True)
                    
# Sort by highest accuracy
allFeaturesPerformanceDF = allFeaturesPerformanceDF.sort_values(by='accuracy', ascending=False)
# Reset the index to maintain a sequential index
allFeaturesPerformanceDF = allFeaturesPerformanceDF.reset_index(drop=True)
pd.set_option('display.max_rows', None)  # Show all rows
print(allFeaturesPerformanceDF.head(100))
# # Scatter plot for all gestures with different colors
# fig_1 = plt.figure(figsize=(10, 8))
# ax_1 = fig_1.add_subplot(projection='3d')

# # Create a color map for the gestures
# colors = {'flat_right_left': 'red', 'vertical_right_left': 'blue', 'up': 'green', 'down': 'purple', 'up&down': 'orange', 'down&up': 'brown', 'flat_left_right': 'pink', 'vertical_left_right': 'gray'}

# # Feature Extraction for all gestures
# min_1, std_1, mean_1 = X_train_1[:, 0], X_train_1[:, 1], X_train_1[:, 2]

# # Plot the figure for all gestures with different colors
# for label in colors:
#     indices = [i for i, x in enumerate(Y_train_1) if x == label]
#     ax_1.scatter(min_1[indices], std_1[indices], mean_1[indices], c=colors[label], label=label)

# # Add axises labels
# ax_1.set_xlabel('Min')
# ax_1.set_ylabel('Max')
# ax_1.set_zlabel('Mean')

# # Add legend
# ax_1.legend()

# #plt.show()


# # Train the KNN model for cell 2
# X_train_flat_2 = np.column_stack((features_solar_2_flat[x_data],features_solar_2_flat[y_data], features_solar_2_flat[z_data]))
# X_train_vertical_2 = np.column_stack((features_solar_2_vertical[x_data], features_solar_2_vertical[y_data], features_solar_2_vertical[z_data]))
# # X_train_up_2 = np.column_stack((features_solar_2_up[x_data],features_solar_2_up[y_data], features_solar_2_up[z_data]))
# # X_train_down_2 = np.column_stack((features_solar_2_down[x_data], features_solar_2_down[y_data], features_solar_2_down[z_data]))
# # X_train_up_down_2 = np.column_stack((features_solar_2_up_down[x_data],features_solar_2_up_down[y_data], features_solar_2_up_down[z_data]))
# # X_train_down_up_2 = np.column_stack((features_solar_2_down_up[x_data], features_solar_2_down_up[y_data], features_solar_2_down_up[z_data]))
# X_train_flat_inverse_2 = np.column_stack((features_solar_2_flat_inverse[x_data],features_solar_2_flat_inverse[y_data], features_solar_2_flat_inverse[z_data]))
# X_train_vertical_inverse_2 = np.column_stack((features_solar_2_vertical_inverse[x_data], features_solar_2_vertical_inverse[y_data], features_solar_2_vertical_inverse[z_data]))
# # X_train_flat_2 = np.column_stack((features_solar_2_flat['min_val'],features_solar_2_flat['max_val'], features_solar_2_flat['mean_val']))
# # X_train_vertical_2 = np.column_stack((features_solar_2_vertical['min_val'], features_solar_2_vertical['max_val'], features_solar_2_vertical['mean_val']))
# X_train_2 = np.concatenate((X_train_flat_2, X_train_vertical_2, X_train_up_2, X_train_down_2, X_train_up_down_2, X_train_down_up_2, X_train_flat_inverse_2, X_train_vertical_inverse_2), axis=0)

# Y_train_2 = ['flat_right_left'] * 40 + ['vertical_right_left']*40 + ['up']*40 + ['down'] * 40 + ['up&down']*40 + ['down&up']*40 + ['flat_left_right']*40 + ['vertical_left_right']*40
# knn_model_2 = KNeighborsClassifier(n_neighbors=3)  
# knn_model_2.fit(X_train_2, Y_train_2)  # train the model

# fig_2 = plt.figure(figsize=(10, 8))
# ax_2 = fig_2.add_subplot(projection='3d')

# # Feature Extraction of cell 2
# min_2_flat, std_2_flat, mean_2_flat = X_train_flat_2[:, 0], X_train_flat_2[:, 1], X_train_flat_2[:, 2]
# min_2_vertical, std_2_vertical, mean_2_vertical = X_train_vertical_2[:, 0], X_train_vertical_2[:, 1], X_train_vertical_2[:, 2]
# # min_2_up, std_2_up, mean_2_up = X_train_up_2[:, 0], X_train_up_2[:, 1], X_train_up_2[:, 2]
# # min_2_down, std_2_down, mean_2_down = X_train_down_2[:, 0], X_train_down_2[:, 1], X_train_down_2[:, 2]
# # min_2_up_down, std_2_up_down, mean_2_up_down = X_train_up_down_2[:, 0], X_train_up_down_2[:, 1], X_train_up_down_2[:, 2]
# # min_2_down_up, std_2_down_up, mean_2_down_up = X_train_down_up_2[:, 0], X_train_down_up_2[:, 1], X_train_down_up_2[:, 2]
# min_2_flat, std_2_flat, mean_2_flat = X_train_flat_inverse_2[:, 0], X_train_flat_inverse_2[:, 1], X_train_flat_inverse_2[:, 2]
# min_2_vertical, std_2_vertical, mean_2_vertical = X_train_vertical_inverse_2[:, 0], X_train_vertical_inverse_2[:, 1], X_train_vertical_inverse_2[:, 2]

# # Plot the figure for cell 2
# ax_2.scatter(min_2_flat, std_2_flat, mean_2_flat, c='r', label='Flat')
# ax_2.scatter(min_2_vertical, std_2_vertical, mean_2_vertical, c='b', label='Vertical')
# # ax_2.scatter(min_2_up, std_2_up, mean_2_up, c='green', label='Up')
# # ax_2.scatter(min_2_down, std_2_down, mean_2_down, c='purple', label='Down')
# # ax_2.scatter(min_2_up_down, std_2_up_down, mean_2_up_down, c='orange', label='Up & Down')
# # ax_2.scatter(min_2_down_up, std_2_down_up, mean_2_down_up, c='brown', label='Down & Up')
# ax_2.scatter(min_2_flat, std_2_flat, mean_2_flat, c='pink', label='Flat Inverse')
# ax_2.scatter(min_2_vertical, std_2_vertical, mean_2_vertical, c='gray', label='Vertical Inverse')

# # Add axises labels
# ax_2.set_xlabel('Min')
# ax_2.set_ylabel('Max')
# ax_2.set_zlabel('Mean')

# ax_2.legend()

# #plt.show()