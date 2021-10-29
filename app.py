from AIKB import Display_Image as display_img
from AIKB import Warp_Image as warp_image
from AIKB import Read_CSV as read_csv

import mediapipe as mp
import numpy as np
import time
import cv2

start = time.time()
mouse_coord = []
click = [0]

#-------------------------------------------------------------------------------------------------------------------------------------------------
# Function to select a different mask
def hoverFunction(event, x, y, flags, param):
    global click
    if event == cv2.EVENT_MOUSEMOVE:
        refPt = [x, y]
        if(mouse_coord):
            mouse_coord.pop()
        mouse_coord.append(refPt)

    if event == cv2.EVENT_LBUTTONDOWN:
        click = [1]

#-------------------------------------------------------------------------------------------------------------------------------------------------

# List the images' paths (PNG files must have transparent backgrounds)
mask_filenames = ['data/batman_1.png','data/batman_2.png',\
                    'data/iron_man_2.png','data/none.png']

# List their corresponding landmark and pixel coordinates
# Landmarks correspond to the value given by MediaPipes library
# Pixel coordinates must match from the images to each landmark 
csv_filenames = ['data/batman_1.csv','data/batman_2.csv',\
                    'data/iron_man_2.csv']

# MediaPipe's functions to extract landmarks' coordinates and
#   face detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,0,0))
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Read mask images to display on output image
mask_width = 70
mask_height = 100
masks_files = []

for file in mask_filenames:
    mask = cv2.imread(file,cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask,(mask_width,mask_height))
    mask = mask / 255.0
    masks_files.append(mask)

# Selection of mask variables
selected = 3
hover = -1

# Define image size parameters and open camera
height,width = 576,768
video = cv2.VideoCapture(0)
ret,image = video.read()
image = cv2.resize(image,(width,height))

# Link the output imshow to the mouse callback function
face_land_img = None
cv2.namedWindow("Live")
cv2.setMouseCallback("Live",hoverFunction)

while(ret):
    frame_start = time.time()

    # Depending on the mask selected, its corresponding image and
    #   landmark location are read
    csv_filename = csv_filenames[0 if selected == 3 else selected]
    img_filename = mask_filenames[selected]
    landmarks,ids,mask_coordinates = read_csv.readCSV(csv_filename)

    # Detection of landmarks
    image = cv2.flip(image,1)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    black_image = np.zeros((height,width,3), np.uint8)
    black_image[::,::] = (230,230,230)

    # Drawing the landmarks on a black image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Save the coordinates of the landmarks of interest
    landmarks_coordinates = []
    for landmark_of_interest in ids:
        x = int(face_landmarks.landmark[landmark_of_interest].x*width)
        y = int(face_landmarks.landmark[landmark_of_interest].y*height)
        landmarks_coordinates.append([x,y])

    # Face detection
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.detections:
        continue

    # Extraction of face location from image
    for detection in results.detections:
        area = detection.location_data.relative_bounding_box
        x1 = int(area.xmin*width)
        y1 = int(area.ymin*height*0.75)
        x2 = x1+int(area.width*width)
        y2 = int(area.ymin*height)+int(area.height*height*1.05)

        face_land_img = black_image[y1:y2,x1:x2]
        face_land_img = cv2.resize(face_land_img,(width//5,height//3))

    # Call warp function to apply homography with the face orientation 
    #   and mask image
    output = warp_image.warpImage(image,landmarks_coordinates,img_filename,\
                                mask_coordinates,selected)

    # Combine results in a single image for output
    result,positions = display_img.displayImage(output,face_land_img,masks_files,selected,hover)
    
    # Pseudo-GUI feature to select the mask of interest
    if(mouse_coord):
        hover = -1
        for i,pos in enumerate(positions):
            y1,y2,x1,x2 = pos
            if(x1 <= mouse_coord[0][0] <= x2 and y1 <= mouse_coord[0][1] <= y2):
                hover = i
                if(click[0]):
                    selected = i
                    click = [0]

    # Display of result and read the next frame
    cv2.imshow("Live",result)
    ret,image = video.read()
    image = cv2.resize(image,(width,height))

    # Press "q" to exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Verify performance of program
    print('\rFPS: {:7.5} Time Elapsed: {:7.5} seconds'.format(1/(time.time()-frame_start),time.time()-start), end='')

print("")
video.release()