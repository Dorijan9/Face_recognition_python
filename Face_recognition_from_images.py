import os
import cv2
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image
import glob
import shutil

def delete_images(directory, prefix, start, end):
    for i in range(start, end + 1):
        file_path = os.path.join(directory, f"{prefix}{i}{end}")
        if os.path.exists(file_path):
            os.remove(file_path)

def copy_images(source_dir, dest_dir, file_pattern):
    for jpgfile in glob.iglob(os.path.join(source_dir, file_pattern)):
        shutil.copy(jpgfile, dest_dir)

def face_recognition_process(num, directory):
    counter_2 = 1
    for i in range(1, num + 1):
        image = face_recognition.load_image_file(os.path.join(directory, f"my_photo-{i}.jpg"))
        face_locations = face_recognition.face_locations(image)

        print(f"I found {len(face_locations)} face(s) in this photograph.")

        for face_location in face_locations:
            top, right, bottom, left = face_location
            print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.show()

            plt.imshow(pil_image)
            plt.savefig(os.path.join(directory, f"result_photo-{counter_2}.png"))
            counter_2 += 1

    return counter_2

def take_photos(counter, wish):
    for g in range(counter, wish + 1):
        print(f"Taking picture number {g}")
        wc = cv2.VideoCapture(0)
        ret, frame = wc.read()
        if ret:
            plt.imshow(frame)
            plt.savefig(f"my_photo-{g}.jpg")
            wc.release()

def continue_taking_photos():
    d = input("Continue taking photos? Enter 'c' for yes, any other letter for no: ")
    return d == "c"

def main():
    counter = 1
    wish = int(input("Enter how many photos do you want to take: "))
    take_photos(counter, wish)

    while continue_taking_photos():
        counter = wish + 1
        wish = int(input("Enter how many more photos do you want to take: ")) + wish
        take_photos(counter, wish)

    file = "/home/pi/facerecognition/script"
    name = "my_photo-"
    name_2 = ".jpg"
    result = "result_photo-"
    result_2 = ".png"
    file_2 = "/home/pi/facerecognition/photos"
    file_3 = "/home/pi/facerecognition/result_photos"
    jpg = "*jpg"
    png = "*.png"
    
    for directory in [file_2, file_3]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    num = face_recognition_process(wish, file)
    copy_images(file, file_2, jpg)
    copy_images(file, file_3, png)
    delete_images(file, wish, name, name_2)
    delete_images(file, num, result, result_2)

if __name__ == "__main__":
    main()
