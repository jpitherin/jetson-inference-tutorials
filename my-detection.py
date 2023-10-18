##docker/run.sh --volume ~/my-detection:/my-detection
##python3 /my-detection/my-detection.py
##xkill -> 

import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("facedetect", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput()

while True:
	img = camera.Capture()
	detections = net.Detect(img)
	
	for detection in detections:
		if net.GetClassDesc(detection.ClassID) == 'person':
			# perform a custom action
			print('detected a person!')
	
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
