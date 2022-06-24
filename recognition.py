#!/usr/in/python3
import jetson.inference
import jetson.utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_URI", type=str, help="URI of the input stream")
parser.add_argument("output_URI", type=str, nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="model to use (see --help)")
opt = parser.parse_args()

input = jetson.utils.videoSource(opt.input_URI)
output = jetson.utils.videoOutput(opt.output_URI)
net = jetson.inference.detectNet(opt.network)
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)

while output.IsStreaming():
        img = input.Capture()
        detections = net.Detect(img)
        for detection in detections:
                print(detection + class_desc + class_idx)
        output.Render(image)
        output.SetStatus("Video Viewer | {:d}x{:d} | {:.1f} FPS".format(image.width, image.height, output.GetFrameRate()))
