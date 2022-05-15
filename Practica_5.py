import cv2
import numpy as np

img = cv2.imread('text.jpg')
rimg = cv2.resize(img,(300,300))
img_g = cv2.cvtColor(rimg,cv2.COLOR_BGR2GRAY)
cv2.imshow('Original',rimg)

ret, thresh1 = cv2.threshold(img_g, 80, 155, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary',thresh1)

ret, thresh2 = cv2.threshold(img_g, 70, 155, cv2.THRESH_BINARY_INV)
cv2.imshow('Threshold Binary Inv',thresh2)

ret, thresh3 = cv2.threshold(img_g, 100, 255, cv2.THRESH_TRUNC)
cv2.imshow('Threshold Trunc',thresh3)

ret, thresh4 = cv2.threshold(img_g, 70, 255, cv2.THRESH_TOZERO)
cv2.imshow('Threshold to Zero',thresh4)

ret, thresh5 = cv2.threshold(img_g, 170, 255, cv2.THRESH_TOZERO_INV)
cv2.imshow('Threshold to Zero Inv',thresh5)

mean = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Mean',mean)

gauss = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Gauss',gauss)

cv2.waitKey(0)
cv2.destroyAllWindows()

ret, otsu = cv2.threshold(img_g,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsu',otsu)

cv2.waitKey(0)
cv2.destroyAllWindows()
