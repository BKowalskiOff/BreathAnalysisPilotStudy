import librosa as lrs
import numpy as np
import os
import threading
import concurrent.futures
import time

def load_audio_as_numpy(audio_path: str, timestamps_path: str) -> np.ndarray:
    # load audio file: arr - ndarray of audio data, sr - sample rate of the audio file
    arr, sr = lrs.load(path=audio_path, sr=None)
    # labels: 1 - inhaling, 2 - exhaling, 0 - none (before first timestamp and after last timestamp from timestamps file)
    labels = np.zeros(arr.shape)
    with open(timestamps_path, 'r') as f:
        inhaling = True
        while True:
            line = f.readline()
            if line == '':
                break
            ts1, ts2, _, _ = line.split()
            # get the indices of array corresponding to timestamps of inhale/exhale start and end
            ts1, ts2 = lrs.time_to_samples([float(ts1), float(ts2)], sr=sr)
            # set appropriate labels
            labels[ts1:ts2] += (1 if inhaling else 2)
            inhaling = not inhaling
    res = np.stack([arr, labels])
    #print(res.shape)
    return res

def preprocess_data(data_path: str):
    train_data = np.ndarray((2,0))
    files = os.listdir(data_path)
    audio_files = [file for file in files if file[-3:] == 'wav']
    timestamp_files = [file for file in files if file[-3:] == 'txt']
    corresponding_files = [(audio_file, timestamp_file) for audio_file in audio_files for timestamp_file in timestamp_files if audio_file[:-3] == timestamp_file[:-3]]
    
    
    for audio_file, timestamp_file in corresponding_files[:100]:
        data = load_audio_as_numpy('\\'.join([data_path, audio_file]), '\\'.join([data_path, timestamp_file]))
        #print(data.shape)
        train_data = np.append(train_data, data, axis=1)
        #print(train_data.shape)
    print(train_data.shape)



def preprocess_data_multithreaded(data_path: str):
    train_data = np.ndarray((2,0))
    files = os.listdir(data_path)
    audio_files = [file for file in files if file[-3:] == 'wav']
    timestamp_files = [file for file in files if file[-3:] == 'txt']
    corresponding_files = [(audio_file, timestamp_file) for audio_file in audio_files for timestamp_file in timestamp_files if audio_file[:-3] == timestamp_file[:-3]]
    
    lock = threading.Lock()
    def load_and_append_multithreaded(audio_path: str, timestamps_path: str):
        nonlocal train_data
        data = load_audio_as_numpy(audio_path, timestamps_path)
        #print(f'processing: ', audio_path, lock.locked(), data.shape)
        with lock:
            #print(train_data.shape)
            train_data = np.append(train_data, data, axis = 1)
            #print(train_data.shape)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        jobs = [executor.submit(load_and_append_multithreaded, '\\'.join([data_path, audio_path]), '\\'.join([data_path, timestamps_path]))
                for audio_path, timestamps_path in corresponding_files[:100]]
        concurrent.futures.wait(jobs)
        print(train_data.shape)


if __name__ == "__main__":
    data_path = '\\'.join([os.getcwd(), 'data'])
    tic = time.perf_counter()
    preprocess_data_multithreaded(data_path)
    toc = time.perf_counter()
    print(f'Time: {toc-tic}s')
    #data = load_audio_as_numpy(audio_path, timestamps_path)
    #print(data.shape)