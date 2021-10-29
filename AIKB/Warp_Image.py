import numpy as np
import cv2

# This function receives the mask file and the locations of the
#	landmarks and warps the given image
#-------------------------------------------------------------------------------------------------------------------------------------------------
def warpImage(image,landmarks_coord,mask_file,mask_coord,selected):
	im_src = cv2.imread(mask_file,cv2.IMREAD_UNCHANGED)
	pts_src = np.array(mask_coord, dtype=float)

	pts_dst = np.array(landmarks_coord, dtype=float)
	h, status = cv2.findHomography(pts_src, pts_dst)
	im_out = cv2.warpPerspective(im_src, h, (image.shape[1],image.shape[0]))

	src = im_out.astype(float)
	src = src / 255.0
	alpha_foreground = src[:,:,3]

	dst = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
	dst = dst.astype(float)
	dst = dst / 255.0

	# Additional code required for blending alpha parameter from mask
	#	image to the live feed, hence the need of transparent backgrounds
	if(selected != 3):
		for color in range(0, 3):
		    dst[:,:,color] = alpha_foreground*src[:,:,color]+\
		    		(1-alpha_foreground)*dst[:,:,color]
	
	dst[:,:,:] = cv2.erode(dst[:,:,:],(5,5),0)
	dst[:,:,:] = cv2.GaussianBlur(dst[:,:,:],(3,3),0)

	return dst

#-------------------------------------------------------------------------------------------------------------------------------------------------