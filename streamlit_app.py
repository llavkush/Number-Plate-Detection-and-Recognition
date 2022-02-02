import streamlit as st
import tensorflow
#import imutils
import cv2
from tensorflow.keras.models import load_model
#from imutils.contours import sort_contours
import numpy as np
import argparse



st.write("""
# Car Number Plate Detection and Recognition
This app recognise the Registraion Number from the Image of the Vehicle Number Plate """)

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
  #img = cv2.imread(image_data,cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY) #convert to grey scale
  gray = cv2.bilateralFilter(gray, 13, 15, 15)
  edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
  contours=cv2.findContours(edged.copy(),cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
  contours = imutils.grab_contours(contours)
  cnts = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
  screenCnt = None
  for c in cnts:
      # approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.018 * peri, True)
      # if our approximated contour has four points, then
      # we can assume that we have found our screen
      if len(approx) == 4:
        screenCnt = approx
        break
  # Masking the part other than the number plate
  mask = np.zeros(gray.shape,np.uint8)
  new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
  new_image = cv2.bitwise_and(image_data,image_data,mask=mask)
  #Cropping Number plate
  (x, y) = np.where(mask == 255)
  (topx, topy) = (np.min(x), np.min(y))
  (bottomx, bottomy) = (np.max(x), np.max(y))
  Cropped = gray[topx:bottomx+1, topy:bottomy+1]
  gray = cv2.bilateralFilter(Cropped, 13, 15, 15) 
  #gray = cv2.cvtColor(Cropped, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 30, 150)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")[0]
  chars = []
  #Image Segentaion for Single Characters
  for c in cnts:
	  # compute the bounding box of the contour
	  (x, y, w, h) = cv2.boundingRect(c)
	  # filter out bounding boxes, ensuring they are neither too small
	  # nor too large
	  if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
		  # extract the character and threshold it to make the character
		  # appear as *white* (foreground) on a *black* background, then
		  # grab the width and height of the thresholded image
		  roi = gray[y:y + h, x:x + w]
		  thresh = cv2.threshold(roi, 0, 255,
			  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		  (tH, tW) = thresh.shape
		  # if the width is greater than the height, resize along the
		  # width dimension
		  if tW > tH:
			  thresh = imutils.resize(thresh, width=32)
		  # otherwise, resize along the height
		  else:
			  thresh = imutils.resize(thresh, height=32)
    # re-grab the image dimensions (now that its been resized)
		# and then determine how much we need to pad the width and
		# height such that our image will be 32x32
		  (tH, tW) = thresh.shape
		  dX = int(max(0, 32 - tW) / 2.0)
		  dY = int(max(0, 32 - tH) / 2.0)
		  # pad the image and force 32x32 dimensions
		  padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
			  left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			  value=(0, 0, 0))
		  padded = cv2.resize(padded, (32, 32))
		  # prepare the padded image for classification via our
		  # handwriting OCR model
		  padded = padded.astype("float32") / 255.0
		  padded = np.expand_dims(padded, axis=-1)
		  # update our list of characters that will be OCR'd
		  chars.append((padded, (x, y, w, h)))
  gray = cv2.bilateralFilter(Cropped, 13, 15, 15)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  padded = cv2.copyMakeBorder(edged, top=dY, bottom=dY,
			left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
			value=(0, 0, 0))
  padded = cv2.resize(padded, (32, 32))
  boxes = [b[1] for b in chars]
  chars = np.array([c[0] for c in chars], dtype="float32")
  #print(chars)
  # OCR the characters using our handwriting recognition model
  preds = loaded_model.predict(chars)
  # define the list of label names
  labelNames = "0123456789"
  labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  labelNames = [l for l in labelNames]
  output=""
  for (pred, (x, y, w, h)) in zip(preds, boxes):
    i = np.argmax(pred)
    prob = pred[i]
    label = labelNames[i]
    output+=label
  cv2.rectangle(Cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
  cv2.putText(Cropped, label, (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
  #cv2_imshow(Cropped)
  cv2.waitKey(0)
  cv2_imshow(Cropped)
  return output





#Loading our model
loaded_model = load_model('OCR_Resnet.h5')


if file is None:
    st.text("Please upload an image file")
else:
    image = cv2.imread(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, loaded_model)
    st.write(prediction)
    
    
    
 
    
st.subheader('Published By Lavkush Gupta')

