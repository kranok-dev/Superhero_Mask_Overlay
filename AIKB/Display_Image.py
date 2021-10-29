import cv2

# This function basically places the landmarks and the mask images
#   for selection
#---------------------------------------------------------------------------------------------
def displayImage(output,face_land_img,masks_files,selected,hover):
    face_land_img = cv2.cvtColor(face_land_img, cv2.COLOR_BGR2BGRA)
    face_land_img = face_land_img / 255.0
    height, width = face_land_img.shape[:2]

    positions = []

    # Place landmark image on the top left corner
    if(face_land_img is not None):
        output[:height,:width] = face_land_img
        output = cv2.rectangle(output, (0,0), (width,height), (0,0,250), 5)

    # Place mask images on the right
    # Depending on the mask selected or hovered over, it shifts it
    #   to the left
    mask_height,mask_width = masks_files[0].shape[:2]
    for i,mask in enumerate(masks_files):
        if(selected == i or hover == i):
            shift = 15
        else:
            shift = 0

        pos_y = [10+i*15+i*mask_height,10+i*15+(i+1)*mask_height]
        pos_x = [output.shape[1]-mask_width-10-shift,output.shape[1]-10-shift]
        positions.append(pos_y+pos_x)
        output[pos_y[0]:pos_y[1],pos_x[0]:pos_x[1]] = mask

        if(selected == i):
            output = cv2.rectangle(output,(pos_x[0],pos_y[0]), \
                                (pos_x[1],pos_y[1]), (0,200,0), 3)
    
    return output,positions

#---------------------------------------------------------------------------------------------