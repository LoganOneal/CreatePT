from enum import Enum
import enum
import time
import threading
import math
import os
import glob
import pathlib
import simpleaudio as sa

from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn

from networktables import NetworkTables
import wpilib
import re
import cv2



import numpy as np

class BlurType(Enum):

    """

    Enums for different blur types

    """

    BOX = 1
    GAUSSIAN = 2
    MEDIAN = 3





"""

definition of all opencv cap props

"""

cap_prop_lookup = {

  "POS_MSEC": cv2.CAP_PROP_POS_MSEC,
  "POS_FRAMES": cv2.CAP_PROP_POS_FRAMES,
  "POS_AVI_RATIO": cv2.CAP_PROP_POS_AVI_RATIO,
  "FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
  "FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
  "FPS": cv2.CAP_PROP_FPS,
  "FOURCC": cv2.CAP_PROP_FOURCC,
  "FRAME_COUNT": cv2.CAP_PROP_FRAME_COUNT,
  "FORMAT": cv2.CAP_PROP_FORMAT,
  "MODE": cv2.CAP_PROP_MODE,
  "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
  "CONTRAST": cv2.CAP_PROP_CONTRAST,
  "SATURATION": cv2.CAP_PROP_SATURATION,
  "HUE": cv2.CAP_PROP_HUE,
  "GAIN": cv2.CAP_PROP_GAIN,
  "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
  "CONVERT_RGB": cv2.CAP_PROP_CONVERT_RGB,
  "RECTIFICATION": cv2.CAP_PROP_RECTIFICATION,

# not supported
#  "WHITE_BALANCE": cv2.CAP_PROP_WHITE_BALANCE,

}

# a list of valid image formats
valid_image_formats = [
    "png",
    "jpg", "jpeg", "jp2", ".jpe",
    "bmp",
    "dib",
    "webp",
    "pbm", "pgm", "ppm",
    "sr", "ras",
    "tiff", "tif"
]

# valid video formats (not official)
valid_video_formats = [
    "mov",
    "avi",
    "mp4",
    "mpeg",
    "flv"
]

class CameraProperties:

    def __init__(self, **kwargs):
        self.props = kwargs
        for x in kwargs:
            if x not in cap_prop_lookup.keys():
                raise KeyError("invalid CAP_PROP: %s" % x)

    def __str__(self):
        ret = "%s(" % (type(self).__name__)
        ss = []
        for p in self.props.keys():
            v = self.props[p]
            if isinstance(v, str):
                v = "'%s'" % v
            ss += ["%s=%s" % (p, v)]
        ret += ",".join(ss)
        ret += ")"
        return ret

    def __getitem__(self, key, default=None):
        return self.props.get(key, default)

    def __setitem__(self, key, val):
        self.props[key] = val


"""

vpl = video pipe line

Think of it similar to how VSTs handle audio, this is a plugin for video

"""

import cv2


class Pipeline:

    def __init__(self, name=None, chain=None):
        self.name = name

        # these are the VPL classes that process the image
        if chain is None:
            self.chain = []
        else:
            self.chain = chain

        self.chain_time = []
        self.chain_fps = (0, [])

        self.vals = {}

        self.is_quit = False


    def __str__(self):
        ret = "Pipeline("

        if self.name != None:
            ret += "name='%s', " % self.name

        ret += "[ \n"


        for vpl in self.chain:
            ret += "  " + str(vpl) + ", \n"
        ret += "])"
        return ret

    def quit(self):
        """

        sets quit flag

        """
        self.is_quit = True


    def add_vpl(self, vpl):
        """

        adds a plugin to the pipeline (at the end), and returns the index in the chain

        """
        self.chain += [vpl]
        return len(self.chain) - 1

    def remove_vpl(self, vpl):
        """

        removes a plugin from the pipeline, or if it is an int, remove that index

        returns either what you passed it, or if it was an int, return the removed plugin

        """
        if isinstance(vpl, int):
            return self.chain.pop(vpl)
        else:
            self.chain.remove(vpl)
            return vpl

    def __raw_chain(self, im, data):
        chain_time = []

        for vpl in self.chain:
            st = time.time()
            im, data = vpl.process(self, im, data)
            et = time.time()
            if self.is_quit:
                break
            chain_time += [et - st]

        def fps(t):
            return 1.0 / t if t != 0 else float('inf')      
        
        self.chain_time = sum(chain_time), chain_time
        self.chain_fps = fps(sum(chain_time)), [fps(i) for i in chain_time]
        
        return im, data, chain_time

    def process(self, image, data=None, loop=False):
        """

        run through the chain, and process the image

        call it with image=None if you use a VideoSource() VPL plugin

        """

        if image is None:
            im = image
        else:
            im = image.copy()


        if data is None:
            data = dict()

        self.is_quit = False
        if loop:
            while not self.is_quit:
                im, data, chain_time = self.__raw_chain(im, data)
                if not self.is_quit:
                    pass
        else:
            im, data, chain_time = self.__raw_chain(im, data)
            if not self.is_quit:
                pass
                
        return im, data

    def __getitem__(self, key, default=None):
        return self.vals.get(key, default)

    get = __getitem__


    def __setitem__(self, key, val):
        self.vals[key] = val


class VPL:
    
    def __init__(self, name=None, **kwargs):
        """

        initialized with a name, and arguments, which you can get later using self["key"], or set using self["key"] = val

        """

        self.name = name
        self.kwargs = kwargs

    def __str__(self):
        ret = "%s(" % (type(self).__name__)

        if self.name != None:
            ret += "name='%s'" % (self.name)

        ss = []
        for k in self.kwargs:
            v = self.kwargs[k]
            if isinstance(v, str):
                v = "'%s'" % v
            
            ss += ["%s=%s" % (k, v)]

        ret += ", ".join(ss)

        ret += ")"
        return ret

    """

    helper methods

    """

    def __getitem__(self, key, default=None):
        return self.kwargs.get(key, default)

    get = __getitem__

    def __setitem__(self, key, val):
        self.kwargs[key] = val


    """

    for async operations, you can call this and it doesn't stall

    """

    def do_async(self, method, args=( )):
        thread = threading.Thread(target=method, args=args)
        thread.daemon = True
        thread.start()

    def process(self, pipe, image, data):
        """

        this is what actually happens to the image (the functionality of the plugin).

          * pipe : the Pipeline() class that called this method (which can be useful)
          * image : the image being processed
          * data : generic data that can be passed between plugins (as a dictionary)

        """

        return image, data

class SubVPL(VPL):
    """

    This is a control VPL, it treats a Pipeline as a single VPL, so you can embed stuff in a VPL

    Usage: SubVPL(pipe=Pipeline(...))

      * "pipe" = pipeline to run as a VPL

    """

    def process(self, pipe, image, data):
        return self["pipe"].process(image, data)


class ForkVPL(VPL):
    """

    This is a control VPL, it forks and runs another Pipeline in another thread.

    This is useful for things that publish to network tables, or look for different vision targets

    Usage: ForkVPL(pipe=Pipeline(...))

      * "pipe" = pipeline to run as a VPL

    THIS ONLY RETURNS THE IMAGE PASSED TO IT

    """

    def process(self, pipe, image, data):
        self.do_async(self["pipe"].process, (image.copy(), data.copy()))
        return image, data


class ForkSyncVPL(VPL):
    """

    This is a control VPL, it forks and runs another Pipeline in another thread.

    This is useful for things that publish to network tables, or look for different vision targets

    Usage: ForkVPL(pipe=Pipeline(...))

      * "pipe" = pipeline to run as a VPL

    THIS ONLY RETURNS THE IMAGE PASSED TO IT

    """

    def process(self, pipe, image, data):
        self["pipe"].process(image.copy(), data.copy())
        return image, data
        


class VideoSource(VPL):
    """

    Usage: VideoSource(source=0)

    optional arguments:

      * "camera" = camera object (default of None)
      * "source" = camera index (default of 0), or a string containing a video file (like "VIDEO.mp4") or a string containing images ("data/*.png")
      * "properties" = CameraProperties() object with CAP_PROP values (see that class for info)
      * "repeat" = whether to repeat the image sequence (default False)
    
    this sets the image to a camera.

    THIS CLEARS THE IMAGE THAT WAS BEING PROCESSED, USE THIS AS THE FIRST PLUGIN

    """

    def camera_single_loop(self):
        self.camera_flag, self.camera_image = self.camera.read()


    def camera_loop(self):
        while True:
            try:
                self.camera_single_loop()
            except:
                pass

    def set_camera_props(self):
        props = self["properties"]
        if props != None:
            for p in props.props:
                #print ("setting: " + str(cap_prop_lookup[p]) + " to: " + str(type(props[p])))
                self.camera.set(cap_prop_lookup[p], props[p])

    def get_camera_image(self):
        return self.camera_flag, self.camera_image

    def get_video_reader_image(self):
        return self.video_reader.read()
    
    def get_image_sequence_image(self):
        my_idx = self.images_idx
        if self.get("repeat", False):
            my_idx = my_idx % len(self.images)
        self.images_idx += 1

        if my_idx >= len(self.images):
            return False, None

        if self.images[my_idx] is None:
            self.images[my_idx] = cv2.imread(self.image_sequence_sources[my_idx])
        return True, self.images[my_idx]

    def process(self, pipe, image, data):
        if not hasattr(self, "has_init"):
            # first time running, default camera
            self.has_init = True
            self.get_image = None

            source = self.get("source", 0)

            # default images
            self.camera_flag, self.camera_image = True, np.zeros((320, 240, 3), np.uint8)

            if isinstance(source, int) or source.isdigit():
                if not isinstance(source, int):
                    source = int(source)
                # create camera
                self.camera = cv2.VideoCapture(source)
                self.do_async(self.camera_loop)
                self.get_image = self.get_camera_image
                self.set_camera_props()
                
            elif isinstance(source, str):
                _, extension = os.path.splitext(source)
                extension = extension.replace(".", "").lower()
                if extension in valid_image_formats:
                    # have an image sequence
                    self.image_sequence_sources = glob.glob(source)
                    self.images = [None] * len(self.image_sequence_sources)
                    self.images_idx = 0
                    self.get_image = self.get_image_sequence_image
                elif extension in valid_video_formats:
                    # read from a video file
                    self.video_reader = cv2.VideoCapture(source)
                    self.get_image = self.get_video_reader_image
                    
            else:
                # use an already instasiated camera
                self.camera = source
                self.do_async(self.camera_loop)
                self.get_image = self.get_camera_image
                self.set_camera_props()
                

        flag, image = self.get_image()

        #data["camera_flag"] = flag
        if image is None:
            pipe.quit()

        return image, data



class VideoSaver(VPL):
    """

    Usage: VideoSaver(path="data/{num}.png")

      * "path" = image format


    optional arguments:
    
      * "every" = save every N frames (default 1 for every frame)

    Saves images as they are received to their destination

    """

    def save_image(self, image, num):
        loc = pathlib.Path(self["path"].format(num="%08d" % num))
        if not loc.parent.exists():
            loc.parent.mkdir(parents=True)
        cv2.imwrite(str(loc), image)

    def process(self, pipe, image, data):
        if not hasattr(self, "num"):
            self.num = 0
        
        if self.num % self.get("every", 1) == 0:
            # async save it
            self.do_async(self.save_image, (image.copy(), self.num))

        self.num += 1

        return image, data


class Resize(VPL):
    """

    Usage: Resize(w=512, h=256)

      * "w" = width, in pixels
      * "h" = height, in pixels

    optional arguments:

      * "method" = opencv resize method, default is cv2.INTER_LINEAR

    """

    def process(self, pipe, image, data):
        height, width, depth = image.shape

        if width != self["w"] or height != self["h"]:
            resize_method = self.get("method", cv2.INTER_LINEAR)
            return cv2.resize(image, (self["w"], self["h"]), interpolation=resize_method), data
        else:
            # if it is the correct size, don't spend time resizing it
            return image, data


class Blur(VPL):
    """

    Usage: Blur(w=4, h=8)

      * "w" = width, in pixels (for guassian blur, w % 2 == 1) (for median blur, this must be an odd integer greater than 1 (3, 5, 7... are good))
      * "h" = height, in pixels (for guassian blur, w % 2 == 1) (for median blur, this is ignored)

    optional arguments:

      * "method" = opencv blur method, default is vpl.BlurType.BOX
      * "sx" = 'sigma x' for the Gaussian blur standard deviation, defaults to letting OpenCV choose based on image size
      * "sy" = 'sigma y' for the Gaussian blur standard deviation, defaults to letting OpenCV choose based on image size

    """

    def process(self, pipe, image, data):
        if self["w"] in (0, None) or self["h"] in (0, None):
            return image, data
        else:
            resize_method = self.get("method", BlurType.BOX)

            if resize_method == BlurType.GAUSSIAN:
                sx, sy = self.get("sx", 0), self.get("sy", 0)
                return cv2.GaussianBlur(image, (self["w"], self["h"]), sigmaX=sx, sigmaY=sy), data
            elif resize_method == BlurType.MEDIAN:
                return cv2.medianBlur(image, self["w"]), data
            else:
                # default is BlurType.BOX
                return cv2.blur(image, (self["w"], self["h"])), data


class Display(VPL):
    """

    Usage: Display(title="mytitle")

        * "title" = the window title

    """

    def process(self, pipe, image, data):

        cv2.imshow(self["title"], image)
        cv2.waitKey(1)

        return image, data


class PrintInfo(VPL):
    """

    Usage: PrintInfo()


    This prints out info about the image and pipeline

    """

    def process(self, pipe, image, data):
        h, w, d = image.shape
        print ("width=%s, height=%s" % (w, h))
        return image, data


class FPSCounter(VPL):
    """

    Usage: FPSCounter()


    Simply adds the FPS in the bottom left corner

    """

    def process(self, pipe, image, data):
        if not hasattr(self, "fps_records"):
            self.fps_records = []

        if not hasattr(self, "last_print"):
            self.last_print = (0, None)

        ctime = time.time()
        self.fps_records += [(ctime, pipe.chain_fps[0])]
        
        # filter only the last second of readings
        self.fps_records = list(filter(lambda tp: abs(ctime - tp[0]) < 1.0, self.fps_records))

        avg_fps = sum([fps for t, fps in self.fps_records]) / len(self.fps_records)

        if self.last_print[1] is None or abs(ctime - self.last_print[0]) > 1.0 / 3.0:
            self.last_print = (ctime, avg_fps)


        font = cv2.FONT_HERSHEY_SIMPLEX
        height, width, _ = image.shape
        geom_mean = math.sqrt(height * width)
        offset = geom_mean * .01
        
        return cv2.putText(image.copy(), "%2.1f" % self.last_print[1], (int(offset), int(height - offset)), font, offset / 6.0, (255, 0, 0), int(offset / 6.0 + 2)), data


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    def update_image(self, image):
        self.RequestHandlerClass.image = image

class MJPGStreamHandle(BaseHTTPRequestHandler):
    """

    handles web requests for MJPG

    """


    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()

        if not hasattr(self, "image"):
            return

        while True:
            # MAKE sure this refreshes the image every time
            im = self.image.copy()

            # encode image
            cv2s = cv2.imencode('.jpg', im)[1].tostring()

            # write a jpg
            self.wfile.write("--jpgboundary".encode())
            self.send_header('Content-type', 'image/jpeg')
            self.send_header('Content-length', str(len(cv2s)).encode())
            self.end_headers()
            self.wfile.write(cv2s)

            while np.array_equal(self.image, im):
                time.sleep(0.01)
        return


class MJPGServer(VPL):
    """

    Usage: MJPGServer(port=5802, fps_cap=None)

      * "port" = the port to host it on

    This is code to host a web server

    This only works on google chrome, connect to "localhost:PORT" to see the image. Or, if you are hosting it on another device (such as a raspi), connect like (raspberrypi.local:PORT) in your browser

    """

    def process(self, pipe, image, data):
        if not hasattr(self, "http_server"):
            self.http_server = ThreadedHTTPServer(('0.0.0.0', self["port"]), MJPGStreamHandle)
            self.http_server.daemon_threads = True
            self.do_async(self.http_server.serve_forever)

        self.http_server.update_image(image.copy())

        return image, data


class ConvertColor(VPL):
    """

    Usage: ConvertColor(conversion=None)

      * conversion = type of conversion (see https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga4e0972be5de079fed4e3a10e24ef5ef0) ex: cv2.COLOR_BGR2HSL

    """

    def process(self, pipe, image, data):
        if self["conversion"] is None:
            return image, data
        else:
            return cv2.cvtColor(image, self["conversion"]), data

class ChannelSplit(VPL):
    """

    Usage: ChannelSplit(store=None)

      * store : the key to store in the data object

    """

    def process(self, pipe, image, data):
        height, width, depth = image.shape

        for i in range(0, depth):
            data[self["store"]][i] = image[:,:,i]
        
        return image, data

class ChannelRecombo(VPL):
    """

    Usage: ChannelSplit(store=None)

      * store : the key to store in the data object

    """

    def process(self, pipe, image, data):
        height, width = data[self["store"]][0].shape
        depth = len(data[self["store"]].keys())
        image = np.zeros((height, width, depth), dtype=data[self["store"]][0].dtype)

        for i in range(0, depth):
            image[:,:,i] = data[self["store"]][i]
        
        return image, data


class InRange(VPL):
    """

    Usage: InRange(H=(20, 40), S=(30, 60), V=(100, 200), mask_key=None)

    """
    def process(self, pipe, image, data):
        H = self.get("H", (0, 157))
        S = self.get("S", (0, 88))
        V = self.get("V", (197, 255))
        mask = cv2.inRange(image, (H[0],S[0],V[0]), (H[1],S[1],V[1]))
        mask_key = self.get("mask_key", None)
        if mask_key is not None:
            data[mask_key] = mask

        return image, data

class ApplyMask(VPL):
    """

    Usage: ApplyMask(mask_key=None)

    """
    def process(self, pipe, image, data):
        mask_key = self.get("mask_key", None)
        if mask_key is not None:
            res = cv2.bitwise_and(image, image, mask=data[mask_key])
            return res, data
        else:
            return image, data

class Erode(VPL):
    """

    Usage: Erode(mask, None, iterations) 
    
    """
    def process(self, pipe, image, data):
        image = cv2.erode(image, None, iterations=2)
        return image, data

class Dilate(VPL):
    """

    Usage: Dilate(mask, None, iterations)

    """
    def process(self, pipe, image, data):
        image = cv2.dilate(image, None, iterations=2)
        return image, data


class FindContours(VPL):
    """ 

    Usage: FindCountours(key="contours")

    """
    def process(self, pipe, image, data):
        # find contours in the mask and initialize the current
        
        cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        data[self["key"]] = []

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ct = 0
            #for c in filter(lambda x: cv2.contourArea(x) > 15, cnts):
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if (radius > 5):
               
                data[self["key"]] += [[center, radius]]
                
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.rectangle(image ,(int(x)-int(radius), int(y)+int(radius)),(int(x)+int(radius), int(y)-int(radius)),(0,255,0),3)
                cv2.putText(image,'LargestContour ' + str(round(x,1)) + ' ' + str(round(y,1)),(int(x)-int(radius), int(y)-int(radius)), cv2.FONT_HERSHEY_PLAIN, 2*(int(radius)/45),(255,255,255),2,cv2.LINE_AA)

                #cv.rectangle(frame , ((int(x)- int(radius), (int(y)+int(radius)) , (1,1),(0,255,0),3)
                cv2.circle(image, center, 5, (255 * (ct % 3 == 2), 255 * (ct % 3 == 1), 255 * (ct % 3 == 0)), -1)
            

                
            ct += 1
        return image, data

class DrawContours(VPL):

    def process(self, pipe, image, data):
        contours = data[self["key"]]
        for center, radius in contours:
            cv2.circle(image, center, 5, (255, 0, 0), -1)
        return image, data


class StoreImage(VPL): 
    
    def process(self, pipe, image, data):
        key = self.get("key", None)
        if key is not None:
            data[key] = image.copy()
        return image, data

class RestoreImage(VPL): 
    
    def process(self, pipe, image, data):
        key = self.get("key", None)
        if key is not None:
            image = data[key]
        return image, data


'''
class DrawMeter(VPL):
    

    Draws a rectangle in the lower left hand corner and scales the x values of the center point. Similar to our previous implementation of light on the robot. 

    

    def process(self, pipe, image, data):
        contours = data[self["key"]]
        h,w,bpp = np.shape(image)
        bar_width = 290
        range_lower = int((w/2)-50)
        range_upper = int((w/2)+50)


        for center, radius in contours:
            x = center[0]
            if x > range_lower and x < range_upper:
                bar_color = (34,139,34)
            else:
                bar_color = (0,0,255)
                

            cv2.rectangle(image, (w-10,h-10), (w-300,h-50) , (0,255,255), cv2.FILLED)
            cv2.rectangle(image,((int(((x/w)*bar_width)+330)),h-5), ((int(((x/w)*bar_width)+350)),h-55), bar_color, cv2.FILLED)

        return image, data
'''

class CoolChannelOffset(VPL):

    def process(self, pipe, image, data):
        h, w, nch = image.shape
        ch = cv2.split(image)
        for i in range(nch):
            xoff = 8 * i
            yoff = 0
            ch[i] = np.roll(np.roll(image[:,:,i], yoff, 0), xoff, 1)
            #image[:,:,i] = np.roll(image[:,:,i], 10, 1)

        image = cv2.merge(ch)

        return image, data

import math

class Bleed(VPL):

    def process(self, pipe, image, data):
        N = self.get("N", 18)
        if not hasattr(self, "buffer"):
            self.buffer = []

        self.buffer.insert(0, image.copy())

        if len(self.buffer) >= N:
            self.buffer = self.buffer[:N]

        #a = [len(self.buffer) - i + N for i in range(0, len(self.buffer))]
        a = [1.0 / (i + 1) for i in range(0, len(self.buffer))]

        # normalize
        a = [a[i] / sum(a) for i in range(len(a))]

        image[:,:,:] = 0

        for i in range(len(a)):
            image = cv2.addWeighted(image, 1.0, self.buffer[i], a[i], 0)

        return image, data


class Pixelate(VPL):

    def process(self, pipe, image, data):
        N = self.get("N", 7.5)

        h, w, d = image.shape

        image = cv2.resize(image, (int(w // N), int(h // N)), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

        return image, data

class Noise(VPL):

    def process(self, pipe, image, data):
        level = self.get("level", .125)

        m = (100,100,100) 
        s = (100,100,100)
        noise = np.zeros_like(image)

        image = cv2.addWeighted(image, 1 - level, cv2.randn(noise, m, s), level, 0)

        return image, data


class DetailEnhance(VPL):

    def process(self, pipe, image, data):
        image = cv2.detailEnhance(image, sigma_s=self.get("r", 10), sigma_r=self.get("s", .15))
        return image, data


class Cartoon(VPL):

    def process(self, pipe, image, data):
        down = self.get("down", 2)
        bilateral = self.get("bilateral", 7)

        for i in range(down):
            image = cv2.pyrDown(image)

        for i in range(bilateral):
            image = cv2.bilateralFilter(image, d=9,
                                    sigmaColor=9,
                                    sigmaSpace=7)

        for i in range(down):
            image = cv2.pyrUp(image)

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_blur = cv2.medianBlur(image_gray, 7)

        image_edge = cv2.adaptiveThreshold(image_blur, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,
                                 blockSize=9,
                                 C=2)

        image_edge = cv2.cvtColor(image_edge, cv2.COLOR_GRAY2RGB)
        image_cartoon = cv2.bitwise_and(image, image_edge)

        return image_cartoon, data




class FindCircles(VPL):

    def process(self, pipe, image, data):
        # detect circles in the image
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 100)
        
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
        
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(image, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        return image, data
 
class Interface(VPL):

    def process(self, pipe, image, data):
        height, width, depth = image.shape

        
        cv2.namedWindow('options',cv2.WINDOW_NORMAL)

        def nothing(x):
            pass

        cv2.createTrackbar('End of Table Left','options',0,50,nothing)
        cv2.createTrackbar('End of Table Right','options',0,50,nothing)
        cv2.resizeWindow('options', 600,100)

        left_bar = cv2.getTrackbarPos('End of Table Left', 'options')
        right_bar = cv2.getTrackbarPos('End of Table Right', 'options')
        
        color = (255,255,0)
        pt1_left = (left_bar*5, 0)
        pt2_left = (left_bar*5, height)

        pt1_right = (width - (right_bar*5), 0)
        pt2_right = (width - (right_bar*5), height)

        cv2.line(image, pt1_left, pt2_left, color, thickness=6, lineType=cv2.LINE_AA, shift=0)
        cv2.line(image, pt1_right, pt2_right, color, thickness=6, lineType=cv2.LINE_AA, shift=0)
            



        return image, data



        

class Score(VPL):

    def __init__(self, **kwargs):
        self.ct = 0
        self.kwargs = kwargs
        #number of frames that must pass for point to be awarded
        self.count = 200

     
    def check(self):
        if self.visible == True:
            self.ct = 0
            print('true')

        elif self.visible == False:
            self.ct = self.ct + 1
            if self.ct > self.count:
                print('point awarded')
                self.ct = 0
            


    def process(self, pipe, image, data):
        self.contours = data[self["key"]]
        if not self.contours:
            self.visible = False

        if self.contours:
            self.visible = True
        
        self.check()
            
        return image, data

        
