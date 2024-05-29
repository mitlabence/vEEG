from pypylon import pylon

Tl_Factory = pylon.TlFactory.GetInstance()
cameras = Tl_Factory.EnumerateDevices()
camera = pylon.InstantCamera(Tl_Factory.CreateDevice(cameras[0]))
camera.Open()

camera.LineSelector.Value = "Line3"  # GPIO for acA1300-75gm
camera.UserOutputSelector.Value = "UserOutput2"
camera.UserOutputValue.Value = True

# UserOutputValueAll Bit-to-Line Association
# Bit 0 is always 0
# Bit 1 configures the status of Line 2
# Bit 2 configures the status of Line 3
# Example: All lines high = 0b110

camera.UserOutputValueAll.Value = 0b100  # only change GPIO, should cause a negative spike
camera.UserOutputValueAll.Value = 0b000  # should cause a positive spike

