# %%
# Use python 3.11.*!
from scipy import signal
import pypylon.pylon as plon
import numpy as np  # conda
import cv2  # pip install opencv-python
from pathlib import Path  # standard lib
import time  # standard lib
import threading  # standard lib
# isntall with pip! Otherwise WASAPI not detected: https://stackoverflow.com/questions/47640188/pyaudio-cant-detect-sound-devices#:~:text=This%20issue%20generally%20arises%20when%20you%20are%20using,pyaudio%20Either%20of%20them%20will%20fix%20your%20issue.
import pyaudio
import librosa  # conda
import queue  # standard lib
import wave  # standard lib
import argparse  # standard lib

import pyfftw
np.fft.fft = pyfftw.interfaces.numpy_fft.fft
np.fft.ifft = pyfftw.interfaces.numpy_fft.ifft
np.fft.fftn = pyfftw.interfaces.numpy_fft.fftn
np.fft.ifftn = pyfftw.interfaces.numpy_fft.ifftn
np.fft.rfft = pyfftw.interfaces.numpy_fft.rfft
np.fft.irfft = pyfftw.interfaces.numpy_fft.irfft
np.fft.rfftn = pyfftw.interfaces.numpy_fft.rfftn
np.fft.irfftn = pyfftw.interfaces.numpy_fft.irfftn
pyfftw.interfaces.cache.enable()

# %%


class Recorder:
    def __init__(self,
                 save_root_path,
                 experiment_name,
                 max_file_length=30,
                 camera_index=0,
                 camera_fps=50,
                 microphone_tag='pett',
                 min_audio_chunk_size=1920,
                 ):
        self.save_root_path = Path(save_root_path)
        self.experiment_name = experiment_name
        self.max_file_length = max_file_length
        self.camera_index = camera_index
        self.camera_fps = camera_fps
        self.microphone_tag = microphone_tag
        self.min_audio_chunk_size = min_audio_chunk_size
        self.logger_queue = queue.PriorityQueue()

    def initialize_start_time(self):
        now = time.localtime()
        start_hh = now.tm_hour
        start_mm = now.tm_min + (2 if now.tm_sec >= 55 else 1)
        if start_mm >= 60:
            start_hh += 1
            start_mm -= 60
        start_time_HH_MM = f"{start_hh:02d}:{start_mm:02d}"
        self.start_time_struct = time.strptime(start_time_HH_MM, "%H:%M")
        self.start_time = time.mktime(time.struct_time((
            time.localtime().tm_year,
            time.localtime().tm_mon,
            time.localtime().tm_mday,
            self.start_time_struct.tm_hour,
            self.start_time_struct.tm_min,
            self.start_time_struct.tm_sec,
            self.start_time_struct.tm_wday,
            self.start_time_struct.tm_yday,
            self.start_time_struct.tm_isdst
        )))
        self.start_time_struct = time.localtime(self.start_time)

    def initialize_save_paths(self):
        self.save_root_path = self.save_root_path
        self.save_root_path.mkdir(parents=True, exist_ok=True)
        self.save_path = self.save_root_path / self.experiment_name
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.video_save_path = self.save_path / \
            f'video_{time.strftime("%Y%m%d_%H%M%S", self.start_time_struct)}'
        self.video_save_path.mkdir(parents=True, exist_ok=True)
        self.single_channel_audio_save_path = self.save_path / \
            f'single_channel_audio_{time.strftime("%Y%m%d_%H%M%S", self.start_time_struct)}'
        self.single_channel_audio_save_path.mkdir(parents=True, exist_ok=True)
        self.log_save_path = self.save_path / \
            f'log_{time.strftime("%Y%m%d_%H%M%S", self.start_time_struct)}.txt'

    def initialize_camera(self):
        self.plon_Tlfactory = plon.TlFactory.GetInstance()
        cameras = self.plon_Tlfactory.EnumerateDevices()
        self.camera = plon.InstantCamera(
            self.plon_Tlfactory.CreateDevice(cameras[self.camera_index]))
        self.camera.Open()
        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRateAbs.SetValue(self.camera_fps)
        self.camera_fps = self.camera.AcquisitionFrameRateAbs.GetValue()
        self.video_height = self.camera.Height.GetValue()
        self.video_width = self.camera.Width.GetValue()
        self.video_pixel_format = self.camera.PixelFormat.GetValue()
        self.video_fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.camera.StartGrabbing(plon.GrabStrategy_LatestImageOnly)
        self.video_frames_queue = queue.PriorityQueue()
        self.cv2_frames_queue = queue.PriorityQueue()

    def video_recorder(self):

        while self.camera.IsGrabbing():
            grabResult = self.camera.RetrieveResult(
                5000, plon.TimeoutHandling_ThrowException)
            current_time = time.time()  # ToDo: use camera timestamp

            if self.camera.IsGrabbing() and grabResult.GrabSucceeded():
                frame = grabResult.Array
                self.video_frames_queue.put((current_time, frame))
            grabResult.Release()

    def video_writer(self):
        file_index = 1
        self.video_output = cv2.VideoWriter(
            str((self.video_save_path / f'T{file_index:07d}.avi').absolute()),
            self.video_fourcc,
            self.camera_fps,
            (self.video_width, self.video_height),
            isColor=False if self.video_pixel_format == 'Mono8' else True
        )
        self.n_recorded_frames = 0
        while self.camera.IsGrabbing() or not self.video_frames_queue.empty():
            if self.video_frames_queue.empty():
                continue
            current_time, frame = self.video_frames_queue.get()
            if self.n_recorded_frames % 50 == 0:  # write to GUI video queue infrequently
                self.cv2_frames_queue.put((current_time, frame.copy()))
            if current_time < self.start_time:
                continue
            self.video_output.write(frame)
            self.n_recorded_frames += 1
            if self.n_recorded_frames >= self.camera_fps * self.max_file_length * 60:
                self.video_output.release()
                file_index += 1
                self.video_output = cv2.VideoWriter(
                    str((self.video_save_path /
                        f'T{file_index:07d}.avi').absolute()),
                    self.video_fourcc,
                    self.camera_fps,
                    (self.video_width, self.video_height),
                    isColor=False if self.video_pixel_format == 'Mono8' else True
                )
                self.n_recorded_frames = 0
        self.video_output.release()

    def initialize_microphone(self):
        self.pyaudio = pyaudio.PyAudio()
        host_api_info = self.pyaudio.get_host_api_info_by_type(
            pyaudio.paWASAPI)
        host_api_index = host_api_info.get('index')
        mic_num = host_api_info.get('deviceCount')
        mic_index = -1
        for i in range(mic_num):
            device_info = self.pyaudio.get_device_info_by_host_api_device_index(
                host_api_index, i)
            if device_info.get('name').lower().find(self.microphone_tag.lower()) != -1:
                mic_index = device_info.get('index')
                break
        if mic_index == -1:
            raise Exception(
                f"Microphone with tag {self.microphone_tag} not found")
        device_info = self.pyaudio.get_device_info_by_index(mic_index)
        self.microphone_sampling_rate = int(
            device_info.get('defaultSampleRate'))
        self.microphone_n_channels = device_info.get('maxInputChannels')
        self.microphone_high_latency = device_info.get(
            'defaultHighInputLatency')
        chunk_duration = np.floor(
            librosa.time_to_samples(self.microphone_high_latency, sr=self.microphone_sampling_rate) / self.min_audio_chunk_size) * \
            librosa.samples_to_time(self.min_audio_chunk_size, sr=self.microphone_sampling_rate
                                    )
        self.audio_chunk_size = librosa.time_to_samples(
            chunk_duration, sr=self.microphone_sampling_rate)
        self.audio_sample_width = self.pyaudio.get_sample_size(pyaudio.paInt16)
        self.audio_format = self.pyaudio.get_format_from_width(
            self.audio_sample_width)
        self.audio_chunks_queue = queue.PriorityQueue()
        self.processing_audio_chunks_queue = queue.PriorityQueue()
        self.audio_stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.microphone_n_channels,
            rate=self.microphone_sampling_rate,
            input=True,
            input_device_index=mic_index,
            frames_per_buffer=self.audio_chunk_size,
            stream_callback=self.audio_callback
        )
        self.audio_stream.start_stream()

    def audio_callback(self, in_data, frame_count, time_info, status):
        # ToDO: use time_info to sync audio and video frames
        current_time = time.time()
        self.audio_chunks_queue.put((current_time, in_data))
        return (None, pyaudio.paContinue)

    def audio_writer(self):
        file_index = 1
        self.audio_output = wave.open(str(
            (self.single_channel_audio_save_path / f'T{file_index:07d}.wav').absolute()), 'wb')
        self.audio_output.setnchannels(self.microphone_n_channels)
        self.audio_output.setsampwidth(self.audio_sample_width)
        self.audio_output.setframerate(self.microphone_sampling_rate)
        self.n_recorded_audio_chunks = 0
        while self.audio_stream.is_active() or not self.audio_chunks_queue.empty():
            if self.audio_chunks_queue.empty():
                continue
            current_time, audio_chunk = self.audio_chunks_queue.get()
            self.processing_audio_chunks_queue.put((current_time, audio_chunk))
            if current_time < self.start_time:
                continue
            self.audio_output.writeframes(audio_chunk)
            self.n_recorded_audio_chunks += 1
            if self.n_recorded_audio_chunks * self.audio_chunk_size >= self.microphone_sampling_rate * self.max_file_length * 60:
                self.audio_output.close()
                file_index += 1
                self.audio_output = wave.open(str(
                    (self.single_channel_audio_save_path / f'T{file_index:07d}.wav').absolute()), 'wb')
                self.audio_output.setnchannels(self.microphone_n_channels)
                self.audio_output.setsampwidth(self.audio_sample_width)
                self.audio_output.setframerate(self.microphone_sampling_rate)
                self.n_recorded_audio_chunks = 0
        self.audio_output.close()

    def initialize_audio_processing(self):
        self.nfft_hop_ratio = 10
        self.nfft = self.audio_chunk_size
        self.hop_length = self.audio_chunk_size // self.nfft_hop_ratio
        self.freqs = librosa.fft_frequencies(
            sr=self.microphone_sampling_rate, n_fft=self.nfft)
        self.cache_size = 1 * self.microphone_sampling_rate
        self.nframes_cache = self.cache_size // self.hop_length
        self.stft_cache = np.zeros((self.nframes_cache, self.freqs.shape[0]))
        self.audio_cache = np.zeros(
            (self.nfft_hop_ratio-1)*self.hop_length+self.audio_chunk_size)

    def audio_processor(self):
        def perform_stft():
            _, _, stft = signal.stft(self.audio_cache, nperseg=self.nfft, noverlap=self.audio_chunk_size -
                                     self.hop_length, nfft=self.nfft, axis=0, padded=False, boundary=None)
            self.stft_cache = np.roll(
                self.stft_cache, -self.nfft_hop_ratio, axis=0)
            np.copyto(
                self.stft_cache[-self.nfft_hop_ratio:, :], np.log10(np.abs(stft)).T)

        while self.audio_stream.is_active() or not self.processing_audio_chunks_queue.empty():
            _, audio_chunk = self.processing_audio_chunks_queue.get()
            audio_chunk = np.frombuffer(audio_chunk, dtype=np.int16)
            self.audio_cache = np.roll(
                self.audio_cache, -self.audio_chunk_size)
            np.copyto(self.audio_cache[-self.audio_chunk_size:], audio_chunk)
            stft_thread = threading.Thread(target=perform_stft)
            stft_thread.start()
            stft_thread.join()

    def logger(self):
        with open(self.log_save_path, 'w') as f:
            while True:
                if self.logger_queue.empty():
                    time.sleep(0.01)
                    continue
                current_time, log = self.logger_queue.get()
                if log is None:
                    break
                f.write(
                    f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))}.{int((current_time - int(current_time)) * 1000):03d} {log}\n')
                f.flush()

    def initialize(self):
        self.initialize_start_time()
        self.initialize_save_paths()
        self.initialize_camera()
        self.initialize_microphone()
        self.initialize_audio_processing()

    def interface(self):
        def add_text(frame, text, font_color='red'):
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (0, 0, 255) if font_color == 'red' else (0, 255, 0)
            lineType = 2
            bottomLeftCornerOfText = (10, 30)
            cv2.putText(frame, text, bottomLeftCornerOfText,
                        font, fontScale, fontColor, lineType)
            return frame

        # frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            'frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        current_time = 0
        video_frame = np.zeros(
            (self.video_height, self.video_width), dtype=np.uint8)
        while self.camera.IsGrabbing() and self.audio_stream.is_active():
            if self.cv2_frames_queue.empty():
                continue
            current_time, video_frame = self.cv2_frames_queue.get()
            video_frame = add_text(
                video_frame, "RECORDING" if current_time >= self.start_time else "NOT RECORDING")/255
            audio_stft_frame = self.stft_cache.copy()
            audio_stft_frame = cv2.resize(
                audio_stft_frame.T[self.freqs <= 125000, :][::-1, :], (self.video_width, self.video_height//4))
            frame = np.concatenate((video_frame, audio_stft_frame), axis=0)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.logger_queue.put((time.time(), None))
                break
            elif key == ord('s'):
                self.logger_queue.put((time.time(), "Trial started"))
            elif key == ord('e'):
                self.logger_queue.put((time.time(), "Event"))
            elif key == ord('f'):
                self.logger_queue.put((time.time(), "Trial ended"))

    def start(self):
        self.video_recorder_thread = threading.Thread(
            target=self.video_recorder)
        self.video_thread = threading.Thread(target=self.video_writer)
        self.audio_thread = threading.Thread(target=self.audio_writer)
        self.audio_processor_thread = threading.Thread(
            target=self.audio_processor)
        self.interface_thread = threading.Thread(target=self.interface)
        self.video_recorder_thread.start()
        self.video_thread.start()
        self.audio_thread.start()
        self.audio_processor_thread.start()
        self.interface_thread.start()
        self.logger_thread = threading.Thread(target=self.logger)
        self.logger_thread.start()

    def stop(self):
        self.interface_thread.join()
        self.camera.Close()
        self.audio_stream.stop_stream()
        self.video_recorder_thread.join()
        self.video_thread.join()
        self.audio_thread.join()
        self.audio_processor_thread.join()
        self.logger_thread.join()
        self.camera.Close()
        self.audio_stream.close()
        self.pyaudio.terminate()

    def run(self):
        self.initialize()
        self.start()
        self.stop()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='recorder_output')
    parser.add_argument('--name', type=str, default='unspecified')
    args = parser.parse_args()
    recorder = Recorder(args.root, args.name)
    recorder.run()
