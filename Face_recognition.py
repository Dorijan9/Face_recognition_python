import face_recognition
import cv2
import numpy as np

def load_known_faces():
    obama_face_encoding = np.load("/home/pi/facerecognition/photos/obama_encoding.npy")
    biden_face_encoding = np.load("/home/pi/facerecognition/photos/biden_encoding.npy")
    trump_face_encoding = np.load("/home/pi/facerecognition/photos/trump_encoding.npy")
    donaj_face_encoding = np.load("/home/pi/facerecognition/photos/donaj_encoding.npy")

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

    return known_face_encodings, known_face_names

def process_frame(frame, known_face_encodings, known_face_names):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_locations, face_names

def main():
    video_capture = cv2.VideoCapture(0)

    known_face_encodings, known_face_names = load_known_faces()

    process_this_frame = True

    for i in range(1, 3):
        ret, frame = video_capture.read()

        face_locations, face_names = process_frame(frame, known_face_encodings, known_face_names)

        if face_locations and face_names:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                frame = cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                cv2.imwrite("result.jpg", frame)
        elif not face_locations and not face_names and i == 2:
            cv2.imwrite("result.jpg", frame)

    video_capture.release()

if __name__ == "__main__":
    main()
