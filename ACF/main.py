import math
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import math

# Dữ liệu này chỉ dùng để plot các kết quả trung gian cho chính xác, không dùng trong thuật toán ACF
voice = [
    [(1.02, 1.88)],
    [(0.53, 1.05)],
    [(0.77, 1.25)],
    [(0.48, 0.77)]
]
unvoice = [
    [(1.88, 1.95)],
    [(1.05, 1.12)],
    [(1.25, 1.27)],
    [(0.45, 0.48)]
]

labels = {
    "voice": voice,
    "unvoice": unvoice
}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def find_median(vector: list) -> float:
    '''
        Hàm tìm trung vị

        Input:
            - vector: mảng các giá trị cần tính trung vị
        Output:
            - Giá trị trung vị
    '''
    n = len(vector)
    vector.sort()

    if n % 2 == 0:
        m1 = vector[n//2]
        m2 = vector[n//2 - 1]
        median = (m1 + m2)/2
        return median
    else:
        median = vector[n//2]
        return median

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def filter_median(vector: list, t: int) -> list:
    '''
        Hàm lọc trung vị của một mảng

        Input:
            - vector: một mảng cần lọc trung vị
            - t: Tìm trung vị của các phần tử thứ i trong đoạn [i-t:i+t]
        Output:
            - một mảng sau khi đã lọc trung vị
    '''
    n = len(vector)

    arr = [0] * n

    for i in range(t):
        arr[i] = vector[i]

    for i in range(t, n - t):
        arr[i] = find_median(vector[i-t:i+t+1])
        
    for i in range(n-t, n):
        arr[i] = vector[i]

    return arr

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def autocorrelation(frame_data: list, F_min: float = None, F_max: float = None, Fs: float = None) -> list:
    '''
        Hàm tương quan

        Input:
            - frame_data: dữ liệu biên độ samples của một frame
            - F_min và F_max: giá trị tần số của giọng người lớn (thường trong đoạn [70Hz, 450Hz])
            - Fs: Tần số của file âm thanh
        Output:
            - List giá trị tương quan tại các độ trễ (lag) = 0..N-1
    '''
    left, right = None, None

    if F_min == None or F_max == None:
        left = 0
        right = len(frame_data)
    else:
        Tmin = 1 / F_max
        Tmax = 1 / F_min
        left = math.floor(Tmin * Fs)
        right = math.floor(Tmax * Fs)

    # normalize input frame_data ==> [-1; 1]
    # Vì dữ liệu biên độ đọc từ file bằng python có giá trị rất lớn, dẫn đến hàm tương quan sẽ tràn số
    # Vì vậy cần chia cho mỗi phần tử của frame cho giá trị lớn nhất trong frame đó để chuẩn hóa
    x = []
    max_magnitude = 0
    for i in range(len(frame_data)):
        if abs(frame_data[i]) > max_magnitude:
            max_magnitude = abs(frame_data[i])

    if (max_magnitude == 0):
        return [0] * len(frame_data)

    for i in frame_data:
        x.append(i / max_magnitude)
    ####

    N = len(x)
    
    xx = []
    for n in range(0, N):
        tmp = 0
        if n == 0 or (n >= left and n <= right):
            for m in range(0, N-n):
                tmp = tmp + x[m] * x[m+n]
        xx.append(tmp)
    
    # Normalize by divide by lag0
    tmp = xx[0]
    for i in range(N):
        xx[i] = xx[i] / tmp

    return xx

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def find_F0(frame_data: list, Fs: float) -> float:
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
    r = math.floor(Tmax * Fs)

    xx = autocorrelation(frame_data, F_min, F_max, Fs)

    max_magnitude = 0
    idx = -1
    for i in range(l, r):
        if xx[i] > xx[i-1] and xx[i] > xx[i+1] and xx[i] > max_magnitude:
            idx = i
            max_magnitude = xx[i]

    F0 = None

    THRESHOLD = 0.45
    if (idx == -1)or (max_magnitude <= THRESHOLD) or abs(frame_data[idx]) <= 100:
        F0 = 0
    else:
        F0 = 1 / (idx * 1 / Fs)

        if (F0 < F_min and F0 > F_max):
            F0 = 0
    
    return F0

def calc(file_path, export_figure_name, FIGURE_DIRECTORY, ANSi, file_idx):
    Fs, data = read(filename=file_path)
    frame_length = 0.03
    number_samples_in_frame = math.floor(frame_length * Fs)
    number_frames = math.floor(len(data) / number_samples_in_frame)

    # F0_data có chiều dài bằng chiều dài của data để lúc plot thì có cùng trục Ox
    F0_data = [0] * len(data)

    # Lưu giữ các giá trị F0 liền kề nhau, phục vụ cho lọc trung vị
    F0_sequence = []
    F0_mid_idx = []

    # Duyệt qua các frame và tính F0
    for frame_idx in range(number_frames):
        start_idx = frame_idx * number_samples_in_frame
        end_idx = (frame_idx + 1) * number_samples_in_frame - 1
        mid_idx = math.floor((start_idx + end_idx) / 2) 
        frame_data = data[start_idx : end_idx]

        F0 = find_F0(frame_data, Fs)
        if F0 != 0:
            F0_sequence.append(F0)
            F0_mid_idx.append(mid_idx)

    # Tinh mean, std cho F0_sequence
    mean = np.mean(F0_sequence)
    std = np.std(F0_sequence)
    print(f"{export_figure_name} Fmean = {mean}, Fstd = {std}")
    print(f">>     diff mean {abs(mean - ANSi[0])}, std {abs(std - ANSi[1])}")

    
    # Lọc trung vị
    F0_sequence = filter_median(F0_sequence, 3)

    for i in range(len(F0_sequence)):
        F0_data[F0_mid_idx[i]] = F0_sequence[i]

    # Plot ra 1 figure gồm 4 hàng, hàng 1,2 dùng để plot dữ liệu trung gian 
    # hàng 3 plot F0 mới tính được, hàng 4 plot dữ liệu đầu vào
    fig = plt.figure(figsize=(25, 20))
    row = 4
    col = 1

    idx_fig = 1
    for key in labels:
        matrix = labels[key][file_idx]

        mid = (matrix[0][0] + matrix[0][1]) / 2
        frame_idx = int(mid / frame_length)
        start_idx = frame_idx * number_samples_in_frame
        end_idx = (frame_idx + 1) * number_samples_in_frame - 1
        
        frame_data = data[start_idx : end_idx]

        # xx = autocorrelation(frame_data, F_min, F_max, Fs)
        xx = autocorrelation(frame_data)

        # Plot kết quả trung gian
        fig.add_subplot(row, col, idx_fig)
        plt.xlabel("sample index")
        plt.ylabel("magnitude")
        plt.title(key.upper())
        plt.plot(xx)
        idx_fig = idx_fig + 1

    fig.add_subplot(row, col, 3)
    plt.xlabel("sample idx")
    plt.ylabel("f0 (Hz)")
    plt.title("F0 FREQUENCY")
    plt.plot(F0_data, ".")

    fig.add_subplot(row, col, 4)
    plt.xlabel("sample index")
    plt.ylabel("magnitude")
    plt.title("ORIGIN FILE")
    plt.plot(data)
    plt.savefig(FIGURE_DIRECTORY + export_figure_name + ".jpg")
    # plt.show()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def main():
    FIGURE_DIRECTORY = "./result/"

    AUDIO_1 = "../audio/audio2/phone_F2.wav"
    AUDIO_2 = "../audio/audio2/phone_M2.wav"
    AUDIO_3 = "../audio/audio2/studio_F2.wav"
    AUDIO_4 = "../audio/audio2/studio_M2.wav"
    AUDIO = [AUDIO_1, AUDIO_2, AUDIO_3, AUDIO_4]

    # Dữ liệu được cung cấp => dùng để tính độ lệch với kết quả thực nghiệm
    # (F0mean, F0std)
    ANS = [(145, 33.7), (129, 18.6), (200, 46.1), (155, 30.8)]

    for i in range(len(AUDIO)):
        calc(AUDIO[i], "result_" + str(i), FIGURE_DIRECTORY, ANS[i], i)

if __name__ == "__main__":
    main()
