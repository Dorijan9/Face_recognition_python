# Open Webcam
# Set PYPLOT environment
# Takes pictures
#Import necessary modules
import cv2
import matplotlib.pyplot as plt
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox

def pyplot_setup():
    plt.close()
    plt.ion()
    fig, ax = plt.subplots()
    return fig, ax

def take_pictures(v, a, c, w) :
    for g in range(c, w + 1):
        if g == c:
            print(f"Taking image number {g}")
            time.sleep(0.001)
            r, f = v.read()
            bbox, label, conf = cv.detect_common_objects(f, confidence=0.1, nms_thresh=0.55, model='yolov3', enable_gpu=False)
            output_image = draw_bbox(f, bbox, label, conf)
            im = a.imshow(output_image)
            plt.pause(0.01)
            plt.show()
            plt.imshow(output_image)
            plt.savefig(f'test-{g}.png')
        else:
            print(f"Taking image number {g}")
            r, f = v.read()
            time.sleep(0.001)
            bbox, label, conf = cv.detect_common_objects(f, confidence=0.1, nms_thresh=0.55, model='yolov3', enable_gpu=False)
            output_image = draw_bbox(f, bbox, label, conf)
            im.set_data(output_image)
            fig.canvas.draw_idle()
            im.autoscale()
            plt.pause(0.01)
            plt.show()
            plt.imshow(output_image)
            plt.savefig(f'test-{g}.png')


#Set up variables
w = cv2.VideoCapture(0)
fig, ax = pyplot_setup()
counter = 1
wish = int(input("Enter how many photos do you want to take: "))
take_pictures(w, ax, counter, wish)

#Create loop if user wants to take more pictures
d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
while d == "c":
    counter = wish + 1
    wish = int(input("Enter how many more photos do you want to take: ")) + wish
    take_pictures(w, ax, counter, wish)
    d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
w.release()
