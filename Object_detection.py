import cv2
import matplotlib.pyplot as plt
import time
import cvlib as cv
from cvlib.object_detection import draw_bbox

def setup_webcam():
    return cv2.VideoCapture(0)

def pyplot_setup():
    plt.close()
    plt.ion()
    fig, ax = plt.subplots()
    return fig, ax

def capture_and_display(v, a, g):
    print(f"Taking image number {g}")
    time.sleep(0.001)
    _, frame = v.read()
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.1, nms_thresh=0.55, model='yolov3', enable_gpu=False)
    output_image = draw_bbox(frame, bbox, label, conf)
    im = a.imshow(output_image)
    plt.pause(0.01)
    plt.show()
    plt.imshow(output_image)
    plt.savefig(f'test-{g}.png')
    return im

def take_pictures(v, a, start_counter, end_counter):
    for g in range(start_counter, end_counter + 1):
        im = capture_and_display(v, a, g) if g == start_counter else update_display(v, im, g)

def update_display(v, im, g):
    print(f"Taking image number {g}")
    _, frame = v.read()
    time.sleep(0.001)
    bbox, label, conf = cv.detect_common_objects(frame, confidence=0.1, nms_thresh=0.55, model='yolov3', enable_gpu=False)
    output_image = draw_bbox(frame, bbox, label, conf)
    im.set_data(output_image)
    plt.pause(0.01)
    plt.show()
    plt.imshow(output_image)
    plt.savefig(f'test-{g}.png')

def main():
    webcam = setup_webcam()
    fig, ax = pyplot_setup()
    start_counter = 1
    wish = int(input("Enter how many photos you want to take: "))
    take_pictures(webcam, ax, start_counter, wish)

    continue_capture = input("Enter 'c' to continue taking photos. Otherwise, press any other key: ")
    while continue_capture.lower() == "c":
        start_counter = wish + 1
        wish = int(input("Enter how many more photos you want to take: ")) + wish
        take_pictures(webcam, ax, start_counter, wish)
        continue_capture = input("Enter 'c' to continue taking photos. Otherwise, press any other key: ")

    webcam.release()

if __name__ == "__main__":
    main()
