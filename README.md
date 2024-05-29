* camera_out_script.py: script testing camera GPIO output
* plon_recorder.py: script that record audio and video, adjusted to our cameras and microphones
* plon_recorder_with_camera_trigger_out.py: similar to plon_recorder.py, except camera also sends spike every N frames on GPIO pin
* tone_ir_trigger.ino: arduino code that periodically triggers simultaneously IR LED + audio + digital spike (for EEG). 
