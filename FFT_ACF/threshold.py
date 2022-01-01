import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import math
from main import FRAME_LENGTH, STE

AUDIO_1_PATH = "../audio/audio3/01MDA.wav"
AUDIO_2_PATH = "../audio/audio3/02FVA.wav"
AUDIO_3_PATH = "../audio/audio3/03MAB.wav"
AUDIO_4_PATH = "../audio/audio3/06FTB.wav"

FILE_PATHS = [AUDIO_1_PATH, AUDIO_2_PATH, AUDIO_3_PATH, AUDIO_4_PATH]


def threshold(f, g):
    # f--> slient STE, g --> voice STE
    # Tim vung giao thoa (overlap) cua f va g
    T_min = max(min(f), min(g))
    T_max = min(max(f), max(g))
    T = 0.5 * (T_min + T_max)

    f = f[f > T_min]
    g = g[g < T_max]

    # Dem so tin hieu < T, > T
    i = sum(f < T)
    p = sum(g > T)

    n = -1
    q = -1

    length_f = len(f)
    length_g = len(g)

    # Tim kiem nhi phan
    while not (i == n and p == q):
        # Kiem tra tong trung binh do lech cua tin hieu > T, < T
        # Thu hep khoang T
        if 1 / length_f * sum(f[f > T] - T) - 1 / length_g * sum(T - g[g < T]) > 0:
            T_min = T
        else:
            T_max = T

        T = 0.5 * (T_min + T_max)

        n = i
        q = p

        i = sum(f < T)
        p = sum(g > T)

    return T


voice = [
    [(0.45, 0.81), (1.53, 1.85), (2.69, 2.86), (3.78, 4.15), (4.84, 5.14)],
    [(0.83, 1.37), (2.09, 2.6), (3.57, 4.0), (4.76, 5.33), (6.18, 6.68)],
    [(1.03, 1.42), (2.46, 2.8), (4.21, 4.48), (6.81, 7.14), (8.22, 8.4)],
    [(1.52, 1.92), (3.91, 4.35), (6.18, 6.6), (8.67, 9.14), (10.94, 11.33)],
]

silence = [
    [
        (0, 0.45),
        (0.81, 1.53),
        (1.85, 2.69),
        (2.86, 3.78),
        (4.15, 4.84),
        (5.14, 5.58),
    ],
    [(0, 0.83), (1.37, 2.09), (2.6, 3.57), (4, 4.76), (5.33, 6.18), (6.68, 7.17)],
    [(0, 1.03), (1.42, 2.46), (2.8, 4.21), (4.48, 6.81), (7.14, 8.22), (8.4, 9.37)],
    [(0, 1.52), (1.92, 3.91), (4.35, 6.18), (6.6, 8.67), (9.14, 10.94), (11.33, 12.75)],
]

labels = {"voice": voice, "silence": silence}
result = {"voice": {"STE": []}, "silence": {"STE": []}}
FRAME_SHIFT = 0.01
for audio_idx in range(len(FILE_PATHS)):
    Fs, data = read(filename=FILE_PATHS[audio_idx])
    data = np.array(data, dtype=float)
    data = data / np.max(abs(data))
    number_samples_in_frame = math.floor(FRAME_LENGTH * Fs)
    number_sample_frame_shift = math.floor(FRAME_SHIFT * Fs)
    # Duyệt qua từng voice/silence của file

    for key in labels:
        matrix = labels[key][audio_idx]
        # Lưu trữ STE của các block trong file
        STE_values = []

        for i in range(len(matrix)):
            start_sample_range = math.floor(matrix[i][0] * Fs)
            end_sample_range = math.floor(matrix[i][1] * Fs)

            start_sample = start_sample_range
            end_sample = start_sample + number_samples_in_frame
            while end_sample < end_sample_range:
                mid_sample = math.floor((start_sample + end_sample) / 2)
                frame_data = data[start_sample:end_sample]
                STE_values.append(STE(frame_data))

                start_sample = (
                    mid_sample
                    + number_sample_frame_shift
                    - number_samples_in_frame // 2
                )
                end_sample = start_sample + number_samples_in_frame

        STE_values = np.array(STE_values)
        result[key]["STE"].append(STE_values)
ste = []
for i in range(len(FILE_PATHS)):
    ste_voice = np.array(result["voice"]["STE"][i])
    ste_silent = np.array(result["silence"]["STE"][i])
    max_val = max(max(abs(ste_voice)), max(abs(ste_silent)))
    ste_voice = ste_voice / max_val
    ste_silent = ste_silent / max_val

    ste.append(threshold(ste_silent, ste_voice))

print(ste)
print(np.mean(ste))
