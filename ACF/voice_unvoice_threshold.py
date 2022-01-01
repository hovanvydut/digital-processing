import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from main import autocorrelation

num = 0

def find_magnitude(frame_data: list, Fs: float) -> float:
    '''
        Tìm F0 của của một frame

        Input:
            - frame_data: dữ liệu biên độ samples của một frame
            - Fs : tần số của file âm thanh (file .wav)
        Output: Tần số F0 của frame
    '''
    F_min = 70
    F_max = 450
    Tmin = 1 / F_max
    Tmax = 1 / F_min
    l = math.floor(Tmin * Fs)
    r = min(math.floor(Tmax * Fs), len(frame_data) - 1)

    xx = autocorrelation(frame_data, F_min, F_max, Fs)

    max_magnitude = 0
    for i in range(l, r):
        if xx[i] > xx[i-1] and xx[i] > xx[i+1] and xx[i] > max_magnitude:
            max_magnitude = xx[i]
    
    return max_magnitude

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

FIGURE_DIRECTORY = "./figure/"

AUDIO_1 = "../audio/audio1/phone_F1.wav"
AUDIO_2 = "../audio/audio1/phone_M1.wav"
AUDIO_3 = "../audio/audio1/studio_F1.wav"
AUDIO_4 = "../audio/audio1/studio_M1.wav"
AUDIO = [AUDIO_1, AUDIO_2, AUDIO_3, AUDIO_4]

voice = [
    [(0.53, 1.14), (1.21, 1.35), (1.45, 1.60), (1.83, 2.20), (2.28, 2.35), (2.40, 2.52), (2.66, 2.73)],
    [(0.46, 1.39), (1.5, 1.69), (1.79, 2.78), (2.86, 2.93), (3.1, 3.29), (3.45, 3.52)],
    [(0.7, 1.1), (1.13, 1.22), (1.27, 1.65), (1.7, 1.76), (1.79, 1.86), (1.92, 2.15)],
    [(0.94, 1.26), (1.33, 1.59), (1.66, 1.78), (1.82, 2.06)]
]
unvoice = [
    [(1.14, 1.21), (1.35, 1.45), (1.60, 1.83), (2.20, 2.28), (2.35, 2.40), (2.52, 2.66), (2.73, 2.75)],
    [(1.39, 1.5), (1.69, 1.79), (2.78, 2.86), (2.93, 3.1), (3.29, 3.45)],
    [(0.68, 0.7), (1.1, 1.13), (1.22, 1.27), (1.65, 1.7), (1.76, 1.79), (1.86, 1.92)],
    [(0.87, 0.94), (1.26, 1.33), (1.59, 1.66), (1.78, 1.82)]
]
silence = [
    [(0.00, 0.53), (2.75, 3.23)],
    [(0, 0.46), (3.52, 4.15)],
    [(0, 0.68), (2.15, 2.86)],
    [(0, 0.87), (2.06, 2.73)]
]

# result = {
#     "voice": [(mean_voice_file0, std_voice_file_0), (mean_voice_file1, std_voice_file_1), ...],
#     "unvoice": ...
# }

result = {
    "voice": [],
    "unvoice": [],
    "silence": []
}

labels = {
    "voice": voice,
    "unvoice": unvoice,
    "silence": silence
}

# CONFIG
frame_length = 0.03 # 20ms

file = open("v_u_s.txt", "w")

# Duyệt qua dữ liệu từng file
for audio_idx in range(len(voice)):

    print("### File #" + str(audio_idx))
    file.write("### File #" + str(audio_idx) + "\n")

    Fs, data = read(filename=AUDIO[audio_idx])
    number_samples_in_frame = math.floor(frame_length * Fs)

    # Duyệt qua từng voice/unvoice/silence của file
    for key in labels:
        print(f"## {key}")
        file.write(f"## {key}\n")

        matrix = labels[key][audio_idx]
        magnitude_global = []

        for i in range(len(matrix)):
            start_time = matrix[i][0]
            end_time = matrix[i][1]

            start_frame_idx = math.floor(start_time / frame_length)
            end_frame_idx = math.floor(end_time / frame_length)

            magnitude_local = []

            for frame_idx in range(start_frame_idx, end_frame_idx+1):
                start_idx = frame_idx * number_samples_in_frame
                end_idx = (frame_idx + 1) * number_samples_in_frame - 1
                
                # print(f"start idx = {start_idx}, end idx = {end_idx}")
                frame_data = data[start_idx : end_idx]
                # print(frame_data)

                magnitude = find_magnitude(frame_data, Fs)

                magnitude_global.append(magnitude)
                magnitude_local.append(magnitude)


            mean = np.mean(magnitude_local)
            std = np.std(magnitude_local)

            tmp_str = f"({start_time}, {end_time}) mean = {mean}, std = {std}"
            print(tmp_str)
            file.write(tmp_str + "\n")

        magnitude_mean = np.mean(magnitude_global)
        magnitude_std = np.std(magnitude_global)
        result[key].append((magnitude_mean, magnitude_std))

        tmp_str = f"magnitude mean = {magnitude_mean}, std = {magnitude_std}"
        print(tmp_str)
        print("\n \n")
        file.write(tmp_str)
        file.write("\n\n")


for key in result:
    arr = result[key]
    mean_arr = []
    std_arr = []

    for i in range(len(arr)):
        mean_arr.append(arr[i][0])
        std_arr.append(arr[i][1])
    
    mean_mean_all = np.mean(mean_arr)
    mean_std_all = np.mean(std_arr)

    str_tmp = f"#### {key}: mean = {mean_mean_all}, {mean_std_all}"
    print(str_tmp)
    file.write(str_tmp + "\n")

file.close()