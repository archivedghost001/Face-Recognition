import cv2
import os
import sys

# Give a way for the image as an arguement
image_path = sys.argv[1]

output_directory = "output/"

os.makedirs(output_directory, exist_ok=True)

# Extract the name file from the image_path
filename = os.path.basename(image_path)

# Separate the name file and extension
name, extension = os.path.splitext(filename)

# Combine the directory output and the name file with the ending "_detected"
output_image_path = os.path.join(
    output_directory, f"{name}_detected{extension}")

# Load the image for the testing
image = cv2.imread(image_path)

# Change the grayish scale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize/Activate the face recognition (cascade default face)
face_cascade = cv2.CascadeClassifier("model-data/face.xml")

# Detecting all faces on the image (variable images)
faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

# Counting the amount of faces detected
face_count = len(faces)


# for every faces will be drawn rectangle
for x, y, width, height in faces:
    cv2.rectangle(
        image,
        (x, y),
        (x + width, y + height),
        color=(0, 128, 0),
        thickness=4,
    )


# print out the amount of detected face
if face_count > 1:
    print(f"{face_count} faces detected on the camera")
else:
    print(f"{face_count} face detected on the camera")

# Setting up the width and height of the image in Windows
width = 720
height = 540

# Setting up the window size based on the original image
cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The results", width, height)

# Showing the image on a new window
cv2.imshow("The results", image)
cv2.waitKey(0)

# Store the image with rectangle
cv2.imwrite(output_image_path, image)
