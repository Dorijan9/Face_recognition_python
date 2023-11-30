def capture_images(video_capture, axis, start_number, total_images, process_frame, known_encodings, known_names):
    for image_number in range(start_number, total_images + 1):
        print(f"Taking image number {image_number}")
        time.sleep(0.001)

        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if face_locations and face_encodings:
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]

                face_names.append(name)

            process_frame = not process_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                frame = cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            display_image(axis, frame)

        elif not face_locations and not face_encodings:
            display_image(axis, frame)

def display_image(axis, image):
    im = axis.imshow(image)
    plt.show()
    plt.pause(0.01)

def main():
    video_capture = cv2.VideoCapture(0)

    known_encodings = [
        np.load("/home/pi/facerecognition/photos/obama_encoding.npy"),
        np.load("/home/pi/facerecognition/photos/biden_encoding.npy"),
        np.load("/home/pi/facerecognition/photos/trump_encoding.npy"),
        np.load("/home/pi/facerecognition/photos/donaj_encoding.npy")
    ]

    known_names = [
        "Barack Obama",
        "Joe Biden",
        "Donald Trump",
        "Dorijan Donaj"
    ]

    fig, ax = pyplot_setup()
    counter = 1
    total_images = int(input("Enter how many photos do you want to take: "))

    capture_images(video_capture, ax, counter, total_images, True, known_encodings, known_names)

    # Create loop if the user wants to take more pictures
    user_input = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")
    while user_input.lower() == "c":
        counter = total_images + 1
        total_images = int(input("Enter how many more photos do you want to take: ")) + total_images
        capture_images(video_capture, ax, counter, total_images, True, known_encodings, known_names)
        user_input = input("Please enter 'c' if you wish to continue taking photos. If not, press any other letter: ")

    # Release handle to the webcam
    video_capture.release()

if __name__ == "__main__":
    main()
