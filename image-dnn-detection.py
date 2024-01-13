import cv2
import numpy as np
import os
import sys


# Pathway for prototxt model Caffe
prototxt_path = "model-data/deploy.prototxt.txt"

# Pathway for model Caffe
model_path = "model-data/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load up model Caffe
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

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
    output_directory, f"{name}_dnn_detected{extension}")

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

# Getting the height and width
height, width = image.shape[:2]

# Pre-procession of image: Resize and Decrease mean
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Sets the image into input nervous system
model.setInput(blob)

# Do the inference and get the result
output = np.squeeze(model.forward())

# Sets up the size of the font and font style
font_scale = 1
font_style = cv2.FONT_HERSHEY_SIMPLEX

# Make a rectangle to detect faces with loops
for i in range(0, output.shape[0]):
    # make a variable trust (kepercayaan) for output looping i
    face_accuracy = output[i, 2]

    # If face accuracy above 50%, then it will make a square aroun dit
    if face_accuracy > 0.5:
        # get coordinate square around and enlarge the size with the original image
        box = output[i, 3:7]*np.array([width, height, width, height])
        # Convert to integer
        start_x, start_y, end_x, end_y = box.astype(np.int64)
        # Draw a rextangle around the face
        cv2.rectangle(
            # rectangle's location, (RGB), and thickness
            image, (start_x, start_y), (end_x, end_y), color=(0, 0, 255), thickness=4,
        )
        # Make a text on the rectangle
        cv2.putText(
            image, f"Warcriminal {face_accuracy*100:.2f}%",
            (start_x, start_y - 5),
            font_style,
            font_scale,
            # blue, green, red
            (0, 0, 255),
            # thickness
            4,
        )

# Print out the amount of faces detected
if face_count > 1:
    print(f"{face_count} Warcriminals detected")
else:
    print(f"{face_count} Warcriminal detected")

# Set up the width and height on the window
width = 720
height = 640

# Set up the window size based on the original image
cv2.namedWindow("The results", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The results", width, height)

# Show the image in a new window
cv2.imshow("The results", image)
cv2.waitKey(0)

# Store the image along with the rectangle
cv2.imwrite(output_image_path, image)
