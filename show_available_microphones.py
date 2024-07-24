import pyaudio
pa = pyaudio.PyAudio()
host_api_info = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
host_api_info.get('index')
n_devices = host_api_info.get('deviceCount')
for i in range(3):
	print(pa.get_device_info_by_host_api_device_index(2, i))