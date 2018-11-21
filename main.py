from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

from io import BytesIO
from time import sleep
from picamera import PiCamera
from PIL import Image
import numpy as np

import Adafruit_ILI9341 as TFT
import Adafruit_GPIO as GPIO
import Adafruit_GPIO.SPI as SPI

#-------------------------------------------------

#yolo = opencvYOLO(modeltype="yolov3-tiny", \
#    objnames="coco.names", \
#    weights="yolov3-tiny.weights",\
#    cfg="yolov3-tiny.cfg")

yolo = opencvYOLO(modeltype="yolov3-tiny", \
    objnames="road.names", \
    weights="yolov3-road-tiny_500000.weights",\
    cfg="yolov3-road-tiny.cfg")


inputType = "webcam"  # webcam, image, video, picam
media = ""
#video_out = "/media/pi/SSD1T/recording/road.avi"
video_out = "record/"
video_length = 300
#--------------------------------------------------

# Raspberry Pi configuration.
DC = 18
RST = 23
SPI_PORT = 0
SPI_DEVICE = 0
# Create TFT LCD display class.
disp = TFT.ILI9341(DC, rst=RST, spi=SPI.SpiDev(SPI_PORT, SPI_DEVICE, max_speed_hz=64000000))
# Initialize display.
disp.begin()


start_time = time.time()

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
        INPUT.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        INPUT.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)
    elif(inputType == "picam"):
        stream = BytesIO()
        camera = PiCamera()
        camera.resolution = (1024, 768)

    if(inputType == "image"):
        yolo.getObject(INPUT, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
        format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        cv2.imshow("Frame", imutils.resize(INPUT, width=850))

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        if(video_out!=""):
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_out + str(time.time()) + ".avi",fourcc, 30.0, (int(width),int(height)))

        frameID = 0
        record_time = time.time()
        while True:
            if(inputType == "picam"):
                camera.capture(stream, format='jpeg', resize=(320, 240))
                stream.seek(0)
                frame = Image.open(stream)
                frame = np.array(frame)
                frame = frame[:, :, ::-1].copy()
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hasFrame = True
            else:
                #INPUT.release()
                #INPUT = cv2.VideoCapture(0)
                #INPUT.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                #INPUT.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                hasFrame, frame = INPUT.read()
                # Stop the program if reached end of video


            if not hasFrame:
                print("Done processing !!!")
                print("--- %s seconds ---" % (time.time() - start_time))
                break

            #yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
            #print ("Object counts:", yolo.objCounts)
            #print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
            #    format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )


            #to TFT
            frame2 = frame.copy()
            frame2 = imutils.rotate_bound(frame2, 270)
            frame2 = imutils.resize(frame2, width=240)
            print(frame2.shape)
            #frame2 = imutils.rotate_bound(frame2, 90)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            frame2 = Image.fromarray(frame2)
            disp.display(frame2)

            if(video_out!=""):
                if(time.time() - record_time > video_length):
                    out.release()
                    out = cv2.VideoWriter(video_out + str(time.time()) + ".avi",fourcc, 30.0, (int(width),int(height)))
                    record_time = time.time()
                else:
                    out.write(frame)

            #k = cv2.waitKey(1)
            #if k == 0xFF & ord("q"):
            #    out.release()
            #    break
