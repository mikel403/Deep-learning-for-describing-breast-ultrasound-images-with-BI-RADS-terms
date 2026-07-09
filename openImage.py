import cv2
def openImage(image_url):
    imag=cv2.imread(image_url,cv2.IMREAD_GRAYSCALE)
    return imag

