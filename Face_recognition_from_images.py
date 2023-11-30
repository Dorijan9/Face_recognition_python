#Import necessary modules
import os
import cv2
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image
import glob
import shutil

def delete_images(f, n, m, mm):
    #Delete all files
    for i in range(1, n + 1):
        if os.path.exists(f"{f}/{m}{i}{mm}"):
            os.remove(f"{f}/{m}{i}{mm}")

def copy_images(f, f2, j):
    #Copy all files
    for jpgfile in glob.iglob(os.path.join(f, j)):
        shutil.copy(jpgfile, f2)

def recognition(n):
    counter_2 = 1
    for i in range(1, n + 1):
        # Load the jpg file into a numpy array
        image = face_recognition.load_image_file(f"my_photo-{i}.jpg")

        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # See also: find_faces_in_picture_cnn.py
        face_locations = face_recognition.face_locations(image)

        print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.show()
            while True:
                plt.imshow(pil_image)
                plt.savefig(f"result_photo-{counter_2}.png")
                counter_2 = counter_2 + 1
                break
    return counter_2

def loop(c, w):
    #Loop that takes picture by picture
    for g in range(c, w + 1):
        print(f"Taking picture number {g}")
        wc = cv2.VideoCapture(0)
        ret, frame = wc.read()
        if ret:
            plt.imshow(frame)
            plt.savefig(f"my_photo-{g}.jpg")
            wc.release()

#Set up variables
counter = 1
wish = int(input("Enter how many photos do you want to take: "))
loop(counter, wish)

#Ask user whether continue taking pictures or continue with processing
d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
while d == "c":
    counter = wish + 1
    wish = int(input("Enter how many more photos do you want to take: ")) + wish
    loop(counter, wish)
    d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")


file = "/home/pi/facerecognition/script"
name = "my_photo-"
name_2 = ".jpg"
result = "result_photo-"
result_2 = ".png"
file_2 = "/home/pi/facerecognition/photos"
file_3 = "/home/pi/facerecognition/result_photos"
jpg = "*jpg"
png = "*.png"
num = recognition(wish)
copy_images(file, file_2, jpg )
copy_images(file, file_3, png)
delete_images(file, wish, name, name_2)
delete_images(file, num, result, result_2)
