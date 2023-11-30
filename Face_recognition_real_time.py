#Import modules
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

#Load an already recognized sample picture
obama_face_encoding = np.load("/home/pi/facerecognition/photos/obama_encoding.npy")
biden_face_encoding = np.load("/home/pi/facerecognition/photos/biden_encoding.npy")
trump_face_encoding = np.load("/home/pi/facerecognition/photos/trump_encoding.npy")
donaj_face_encoding = np.load("/home/pi/facerecognition/photos/donaj_encoding.npy")

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    trump_face_encoding,
    donaj_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Donald Trump",
    "Dorijan Donaj"
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def pyplot_setup():
    plt.close()
    plt.ion()
    fig, ax = plt.subplots()
    return fig, ax

def take_pictures(v, a, c, w, p, e, n, face_locations, face_encodings, face_names):

    for g in range(c, w + 1):
        if g == c:
            print(f"Taking image number {g}")
            time.sleep(0.001)
            # Grab a single frame of video
            ret, frame = v.read()

            for i in range(1, 3):
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for j in range(1, 3):
                    if face_locations and face_encodings:
                        face_names = []
                        for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                            matches = face_recognition.compare_faces(e, face_encoding)
                            name = "Unknown"

                            # # If a match was found in known_face_encodings, just use the first one.
                            # if True in matches:
                            #     first_match_index = matches.index(True)
                            #     name = known_face_names[first_match_index]

                            # Or instead, use the known face with the smallest distance to the new face
                            face_distances = face_recognition.face_distance(e, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = n[best_match_index]

                            face_names.append(name)

                        p = not p

                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            frame = cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            im = a.imshow(frame)
                            plt.show()
                            plt.pause(0.01)

                    elif not face_locations and not face_encodings and j == 2:
                        im = a.imshow(frame)
                        plt.show()
                        plt.pause(0.01)


        else:
            print(f"Taking image number {g}")
            time.sleep(0.001)
            # Grab a single frame of video
            ret, frame = video_capture.read()

            for i in range(1, 3):
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for j in range(1, 3):
                    if face_locations and face_encodings:
                        face_names = []
                        for face_encoding in face_encodings:
                            # See if the face is a match for the known face(s)
                            matches = face_recognition.compare_faces(e, face_encoding)
                            name = "Unknown"

                            # # If a match was found in known_face_encodings, just use the first one.
                            # if True in matches:
                            #     first_match_index = matches.index(True)
                            #     name = known_face_names[first_match_index]

                            # Or instead, use the known face with the smallest distance to the new face
                            face_distances = face_recognition.face_distance(e, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = n[best_match_index]

                            face_names.append(name)

                        p = not p

                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, face_names):
                            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            frame = cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                            im.set_data(frame)
                            fig.canvas.draw_idle()
                            im.autoscale()
                            plt.show()
                            plt.pause(0.01)

                    elif not face_locations and not face_encodings and j == 2:
                        im.set_data(frame)
                        fig.canvas.draw_idle()
                        im.autoscale()
                        plt.show()
                        plt.pause(0.01)


#Activate functions
fig, ax = pyplot_setup()
counter = 1
wish = int(input("Enter how many photos do you want to take: "))
take_pictures(video_capture, ax, counter, wish, process_this_frame, known_face_encodings, known_face_names, face_locations, face_encodings, face_names)

#Create loop if user wants to take more pictures
d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
while d == "c":
    counter = wish + 1
    wish = int(input("Enter how many more photos do you want to take: ")) + wish
    take_pictures(video_capture, ax, counter, wish, process_this_frame, known_face_encodings, known_face_names, face_locations, face_encodings, face_names)
    d = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
# Release handle to the webcam
video_capture.release()
