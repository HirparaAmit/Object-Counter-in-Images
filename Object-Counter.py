import cv2
from cv2 import resize
import numpy as np

# created function which take image and number of colors as input and return the image after applying K-Means clustering.
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

#load image
image = cv2.imread('C:/Users/admin/Downloads/Internship/2/images/count=6.jpeg')
# resize image
image = resize(image, (640,480))
#call the K-Means clustering function
image = quantimage(image, 15)

#convert image into canny mode
canny = cv2.Canny(image, 550,50)
#blur the canny image
blur = cv2.GaussianBlur(canny, (25,25), 3)

#create contours
(cnt, hierarchy) = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, cnt, -1, (0,0,255), 2)

text = f"No. of Objects = {len(cnt)}"
#put text on image
cv2.putText(image, text, (10,30), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255))
cv2.imshow("Output", image)
cv2.waitKey(0)