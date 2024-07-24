from pypylon import pylon

tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()
for i_device, device in enumerate(devices):
    print(f"Device #{i_device}: {device.GetFriendlyName()}")
