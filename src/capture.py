import cv2
import argparse
import vpl
from networktables import NetworkTables
import socket


parser = argparse.ArgumentParser(description='Example webcam view for punkvision')

parser.add_argument('--source', type=str, default="0", help='camera number, image glob (like "data/*.png"), or video file ("x.mp4")')
parser.add_argument('--size', type=int, nargs=2, default=(640, 480), help='image size')
parser.add_argument('--blur', type=int, nargs=2, default=(12,12), help='blur size')

parser.add_argument('--save', default=None, help='save the stream to files (ex: "data/{num}.png"')

parser.add_argument('--stream', default=None, type=int, help='stream to a certain port (ex: "5802" and then connect to "localhost:5802" to view the stream)')

parser.add_argument('--noshow', action='store_true', help="if this flag is set, do not show a window (useful for raspberry PIs without a screen, you can use --stream)")

parser.add_argument('--noprop', action='store_true', help="if this flag is set, do not set capture properties (some may fail without this, use this if you are getting a QBUF error)")

args = parser.parse_args()


pipe = vpl.Pipeline("process")
fork = vpl.Pipeline("record")
cam_props = vpl.CameraProperties()

# set preferred width and height
if not args.noprop:
    cam_props["FRAME_WIDTH"] = args.size[0]
    cam_props["FRAME_HEIGHT"] = args.size[1]


# find the source
pipe.add_vpl(vpl.VideoSource(source=args.source, properties=cam_props))

pipe.add_vpl(vpl.ForkSyncVPL(pipe=fork))
#fork.add_vpl(vpl.ShowGameInfo())


#fork.add_vpl(vpl.Resize(w=480, h=320))

# resize
pipe.add_vpl(vpl.Resize(w=args.size[0], h=args.size[1]))

#blur
pipe.add_vpl(vpl.Blur(w=args.blur[0], h=args.blur[1], method=vpl.BlurType.BOX))

#convert to HSV
pipe.add_vpl(vpl.ConvertColor(conversion=cv2.COLOR_BGR2HSV))

#Filter HSV threshold
pipe.add_vpl(vpl.InRange(mask_key="mask"))
pipe.add_vpl(vpl.ApplyMask(mask_key="mask"))

#Erode
pipe.add_vpl(vpl.StoreImage(key="normal"))
pipe.add_vpl(vpl.RestoreImage(key="mask"))
pipe.add_vpl(vpl.Erode())

#Dilate
pipe.add_vpl(vpl.Dilate())

#Find Contours
pipe.add_vpl(vpl.FindContours(key="contours"))

pipe.add_vpl(vpl.RestoreImage(key="normal"))

#Convert back to BGR
pipe.add_vpl(vpl.ConvertColor(conversion=cv2.COLOR_HSV2BGR))

#Draws dot on center point of convex hull
pipe.add_vpl(vpl.DrawContours(key="contours"))

pipe.add_vpl(vpl.DrawMeter(key="contours"))

#use score module
#pipe.add_vpl(vpl.Score(key="contours"))




# add a FPS counter
pipe.add_vpl(vpl.FPSCounter())



#stream it
if not args.noshow:
    pipe.add_vpl(vpl.Display(title="footage from " + str(args.source)))
    fork.add_vpl(vpl.Display(title="fork"))
if args.stream is not None:
    fork.add_vpl(vpl.MJPGServer(port=args.stream))

try:
      # we let our VideoSource do the processing, autolooping
      pipe.process(image=None, data=None, loop=True)
except (KeyboardInterrupt, SystemExit):
    print("keyboard interrupt, quitting")
    exit(0)