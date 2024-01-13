import cv2

# initialize face detector
face_cascade = cv2.CascadeClassifier(
    "model-data/face.xml")

# make new camera object
capture = cv2.VideoCapture(0)

# Infinite loop of death
while True:
    # reads the image/frame from the camera
    live, camera = capture.read()
    # convert the image from grayscale
    image_gray = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)

    # detecting all faces on the camera
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    # count the amount of faces detected
    face_count = len(faces)

    # for every face, draw rectangle/square
    for x, y, width, height in faces:
        cv2.rectangle(
            camera, (x, y), (x + width, y + height), color=(0, 128, 0), thickness=3
        )
    # shows new window
    cv2.imshow("Face detect v5.1", camera)

    # print out the amount of face detected
    if face_count > 1:
        print(f"{face_count} faces detected on the camera", end="\r")
    else:
        print(f"{face_count} face detected on camera", end="\r")

    # if the user presses "q", the loop will stop
    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
