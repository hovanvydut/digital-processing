import numpy as np
from scipy.io.wavfile import read
from scipy.fft import fft
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import math
import re


def STE(frame_data):
    """
    Short-Time Energy(STE)
    Input:
        - frame_data (đã được chuẩn hóa)
    Output:
        - STE của frame
    """
    x = frame_data
    N = len(x)

    # caculate STE
    STEn = 0
    for m in range(0, N):
        STEn = STEn + x[m] ** 2

    return STEn


def g_composite(f, T):
    g = np.zeros_like(f)
    fmax = max(f)
    fmin = min(f)
    for i in range(len(g)):
        if f[i] >= T:
            g[i] = (f[i] - T) / (fmax - T)
        else:
            g[i] = (f[i] - T) / (T - fmin)
    return g


def find_voice(file_path, T_STE, FRAME_LENGTH):
    # Đọc dữ liệu từ file, Fs là tần số lấy mẫu
    Fs, data = read(filename=file_path)
    data = np.array(data, dtype=float)

    # Chuẩn hóa dữ liệu đọc được từ file tín hiệu về [-1, 1]
    data = data / np.max(abs(data))

    # Chia data thành các frame
    number_samples_in_frame = math.floor(FRAME_LENGTH * Fs)
    number_sample_frame_shift = math.floor(FRAME_SHIFT * Fs)
    # Biến lưu trữ giá trị STE cho từng frame trong file tín hiệu
    # STE_values = np.zeros(number_frames, dtype=float)
    STE_values = np.zeros_like(data)
    # STE_values[:] = -1

    start_sample_range = 0
    end_sample_range = data.size
    start_sample = start_sample_range
    end_sample = start_sample + number_samples_in_frame
    while end_sample < end_sample_range:
        mid_sample = math.floor((start_sample + end_sample) / 2)

        frame_data = data[start_sample:end_sample]
        STE_values[start_sample:end_sample] = STE(frame_data)

        start_sample = (
            mid_sample + number_sample_frame_shift - number_samples_in_frame // 2
        )
        end_sample = start_sample + number_samples_in_frame
    STE_values[start_sample : len(data)] = STE(data[start_sample : len(data)])

    # Chuẩn hóa về [-1, 1]
    STE_values = STE_values / np.max(abs(STE_values))

    # Hàm g(f)
    # g_STE = g_composite(STE_values, T_STE)

    # Decision Making
    VU = STE_values >= T_STE

    for i in range(len(VU) - 1):
        if VU[i] == False and VU[i + 1] == True:
            start = i
            while i < len(VU) - 1 and not (VU[i] == True and VU[i + 1] == False):
                i += 1
            end = i
            if end - start < 0.2 * Fs:
                VU[start : end + 1] = False

    VU2 = VU

    arr = []
    for i in range(len(VU2) - 1):
        if VU2[i] != VU2[i + 1]:
            arr.append(i)

    set_positive = set()
    for i in range(len(arr) - 1):
        if arr[i + 1] - arr[i] > math.floor(0.2 * Fs):
            set_positive.add(arr[i])
            set_positive.add(arr[i + 1])

    set_positive = list(set_positive)
    set_positive.sort()
    tmp = []
    for i in range(len(set_positive)):
        tmp.append(set_positive[i] / Fs)
    result = []
    for i in range(0, len(tmp), 2):
        result.append((tmp[i], tmp[i + 1]))

    return result, VU2, STE_values


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def find_median(vector: list) -> float:
    """
    Hàm tìm trung vị

    Input:
        - vector: mảng các giá trị cần tính trung vị
    Output:
        - Giá trị trung vị
    """
    n = len(vector)
    vector.sort()

    if n % 2 == 0:
        m1 = vector[n // 2]
        m2 = vector[n // 2 - 1]
        median = (m1 + m2) / 2
        return median
    else:
        median = vector[n // 2]
        return median


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def filter_median(vector: list, t: int) -> list:
    """
    Hàm lọc trung vị của một mảng

    Input:
        - vector: một mảng cần lọc trung vị
        - t: Tìm trung vị của các phần tử thứ i trong đoạn [i-t:i+t]
    Output:
        - một mảng sau khi đã lọc trung vị
    """
    n = len(vector)

    arr = [0] * n

    for i in range(t):
        # arr[i] = vector[i]
        arr[i] = find_median(vector[i : i + 2 * t])

    for i in range(t, n - t):
        arr[i] = find_median(vector[i - t : i + t + 1])

    for i in range(n - t, n):
        # arr[i] = vector[i]
        arr[i] = find_median(vector[i - 2 * t : i + 1])

    return arr


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def autocorrelation(
    frame_data: list,
    spectral_resolution,
    F_min: float = None,
    F_max: float = None,
    Fs: float = None,
) -> list:
    """
    Hàm tương quan

    Input:
        - frame_data: dữ liệu biên độ samples của một frame
        - F_min và F_max: giá trị tần số của giọng người lớn (thường trong đoạn [70Hz, 450Hz])
        - Fs: Tần số của file âm thanh
    Output:
        - List giá trị tương quan tại các độ trễ (lag) = 0..N-1
    """

    # normalize input frame_data ==> [-1; 1]
    # Vì dữ liệu biên độ đọc từ file bằng python có giá trị rất lớn, dẫn đến hàm tương quan sẽ tràn số
    # Vì vậy cần chia cho mỗi phần tử của frame cho giá trị lớn nhất trong frame đó để chuẩn hóa
    x = frame_data / max(abs(frame_data))
    N = len(x)

    xx = []
    for n in range(0, N):
        tmp = 0
        for m in range(0, N - n):
            tmp = tmp + x[m] * x[m + n]
        xx.append(tmp)

    xx = np.array(xx)
    xx = xx / xx[0]

    return xx


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def find_F0(frame_data: list, Fs: float, spectral_resolution) -> float:
    """
    Tìm F0 của của một frame

    Input:
        - frame_data: dữ liệu biên độ samples của một frame
        - Fs : tần số của file âm thanh (file .wav)
        - spectral_resolution: độ phân giải biên độ của phổ
    Output: Tần số F0 của frame
    """
    F_min = 70
    F_max = 400
    l = math.floor(F_min / spectral_resolution)
    r = math.floor(F_max / spectral_resolution)

    xx = autocorrelation(frame_data, spectral_resolution, F_min, F_max, Fs)

    max_magnitude = 0
    max_idx = -1
    for i in range(l, r):
        if xx[i] > xx[i - 1] and xx[i] > xx[i + 1] and xx[i] > max_magnitude:
            max_idx = i
            max_magnitude = xx[i]

    F0 = None

    if max_idx == -1:
        F0 = 0
    else:
        F0 = spectral_resolution * max_idx

        if F0 < F_min and F0 > F_max:
            F0 = 0

    return F0


# Global variable

FRAME_LENGTH = 0.02
T_STE = 0.0025
# T_STE = 0.002
FRAME_SHIFT = 0.01

# AUDIO_1_PATH = "../audio/audio3/01MDA.wav"
# AUDIO_2_PATH = "../audio/audio3/02FVA.wav"
# AUDIO_3_PATH = "../audio/audio3/03MAB.wav"
# AUDIO_4_PATH = "../audio/audio3/06FTB.wav"

AUDIO_1_PATH = "../audio/audio4/30FTN.wav"
AUDIO_2_PATH = "../audio/audio4/42FQT.wav"
AUDIO_3_PATH = "../audio/audio4/44MTT.wav"
AUDIO_4_PATH = "../audio/audio4/45MDV.wav"

FILE_PATHS = [AUDIO_1_PATH, AUDIO_2_PATH, AUDIO_3_PATH, AUDIO_4_PATH]

# expected_voice_times = [
#     [(0.45, 0.81), (1.53, 1.85), (2.69, 2.86), (3.78, 4.15), (4.84, 5.14)],
#     [(0.83, 1.37), (2.09, 2.6), (3.57, 4.0), (4.76, 5.33), (6.18, 6.68)],
#     [(1.03, 1.42), (2.46, 2.8), (4.21, 4.52), (6.81, 7.14), (8.22, 8.5)],
#     [(1.52, 1.92), (3.91, 4.35), (6.18, 6.6), (8.67, 9.14), (10.94, 11.33)],
# ]

expected_voice_times = [
    [(0.59, 0.97), (1.76, 2.11), (3.44, 3.77), (4.7, 5.13), (5.96, 6.28)],
    [(0.46, 0.99), (1.56, 2.13), (2.51, 2.93), (3.79, 4.38), (4.77, 5.22)],
    [(0.93, 1.42), (2.59, 3), (4.71, 5.11), (6.26, 6.66), (8.04, 8.39)],
    [(0.88, 1.34), (2.35, 2.82), (3.76, 4.13), (5.04, 5.5), (6.41, 6.79)],
]

# EXPECTED_RESULT = [(135.5, 5.4), (239.7, 5.6), (115, 4.5), (202.9, 15.5)]
EXPECTED_RESULT = [(233.2, 11.6), (242.7, 8.5), (125.7, 8.5), (177.8, 5.7)]

if __name__ == "__main__":

    for i in range(len(FILE_PATHS)):
        file_path = FILE_PATHS[i]
        match = re.search("\\w*(?=\\.wav)", file_path)
        subtitle = file_path[match.span()[0] : match.span()[1]]

        Fs, data = read(file_path)

        data = np.array(data)
        # Chuan hoa data
        data = data / max(abs(data))

        expected_voice = expected_voice_times[i]

        # # # # # # # # # # # # # # # # # STE # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        voices, x_VU, x_axis_STE = find_voice(file_path, T_STE, FRAME_LENGTH)

        # # # # # # # # # # # # # # # # # FFT # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        FFT_POINT = 8192 * 2

        spectral_resolution = Fs / FFT_POINT
        t = np.arange(0, len(data) / Fs, 1 / Fs)

        number_samples_in_frame = math.floor(FRAME_LENGTH * Fs)
        number_sample_frame_shift = math.floor(FRAME_SHIFT * Fs)
        data_raw_to_plot = None
        data_acf_to_plot = None
        data_fft_to_plot = None
        F0_values = []
        F0_idx = []

        count = 0
        for voice in voices:
            start_sample_range = math.floor(voice[0] * Fs)
            end_sample_range = math.floor(voice[1] * Fs)

            start_sample = start_sample_range
            end_sample = start_sample + number_samples_in_frame

            while end_sample < end_sample_range:
                count += 1
                mid_sample = math.floor((start_sample + end_sample) / 2)
                frame_data = data[start_sample:end_sample]
                y = abs(fft(frame_data, n=FFT_POINT))

                F0 = find_F0(y[:500], Fs, spectral_resolution)
                if F0 != 0:
                    F0_values.append(F0)
                    F0_idx.append(mid_sample)

                if count == 13:
                    F_min = 70
                    F_max = 400
                    l = math.floor(F_min / spectral_resolution)
                    r = math.floor(F_max / spectral_resolution)
                    data_raw_to_plot = frame_data
                    data_fft_to_plot = y
                    data_acf_to_plot = autocorrelation(
                        y[:500], spectral_resolution, F_min, F_max, Fs
                    )

                start_sample = (
                    mid_sample
                    + number_sample_frame_shift
                    - number_samples_in_frame // 2
                )
                end_sample = start_sample + number_samples_in_frame

        # Filter
        F0_values = filter_median(F0_values, 5)
        for j in range(3, len(F0_values)):
            if (
                (F0_values[j] - F0_values[j - 1] > 50)
                and (F0_values[j] - F0_values[j - 2] > 50)
                and (F0_values[j] - F0_values[j - 3] > 50)
            ):
                F0_values[j] = F0_values[j - 1]

        F0_data = np.zeros_like(data)
        for j in range(len(F0_values)):
            F0_data[F0_idx[j]] = F0_values[j]

        delay_time = 0
        total_time = 0
        for j in range(len(expected_voice)):
            delay_time += abs(expected_voice[j][0] - voices[j][0]) + abs(
                expected_voice[j][1] - voices[j][1]
            )
            total_time += expected_voice[j][1] - expected_voice[j][0]

        print("🔥  " + subtitle + "   🔥")
        F0_mean = round(np.mean(F0_values), 2)
        F0_std = round(np.std(F0_values), 2)
        print(f"F0: mean = {F0_mean}, std = {F0_std}")
        print(
            f"🎄 delta-mean = {round(abs(F0_mean-EXPECTED_RESULT[i][0]), 2)}, delta-std={round(abs(F0_std-EXPECTED_RESULT[i][1]), 2)}"
        )
        print(
            # f"total delay time: {round(delay_time, 2)}, % delay time = {round(delay_time/total_time,2)}"
            f"total delay time: {round(delay_time, 2)}"
        )
        print("")

        fig, axs = plt.subplots(4, figsize=(30, 25))
        axs[0].plot(data)
        axs[0].plot(x_VU, color="r")
        axs[0].plot(x_axis_STE)
        axs[0].legend(["signal", "STE >= T_STE", "STE"])
        axs[0].set_title("intermediate result 1- STE")
        axs[0].set(xlabel="sample index", ylabel="normalized magnitude")

        axs[1].plot(data_acf_to_plot)
        axs[1].set_title("intermediate result 2- FFT")
        axs[1].set(xlabel="sample index", ylabel="normalized magnitude")

        axs[2].plot(F0_data, ".")
        axs[2].set(xlabel="sample index", ylabel="Hz")

        axs[3].plot(data)
        for i in range(len(expected_voice)):
            start_time, end_time = expected_voice[i]
            idx1 = math.floor(start_time * Fs)
            idx2 = math.floor(end_time * Fs)
            axs[3].axvline(x=idx1, color="r", alpha=0.5)
            axs[3].axvline(x=idx2, color="r", alpha=0.5)
        for i in range(len(x_VU) - 2):
            if x_VU[i] != x_VU[i + 1]:
                axs[3].axvline(x=i, color="g", alpha=0.5)
        handles = [
            Line2D([0], [0], label="signal", color="b"),
            Line2D([0], [0], label="provided", color="r"),
            Line2D([0], [0], label="caculated", color="g"),
        ]
        axs[3].legend(handles=handles)
        axs[3].set(xlabel="sample index", ylabel="normalized magnitude")

        fig.suptitle(subtitle)
        plt.savefig(f"./result/{subtitle}.jpg")
        plt.show(block=False)

    print("🎅⛄🛷🦌🌟🔔❄")
    # block chuong trinh de hien thi figures
    plt.show()
