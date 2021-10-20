import cv2
import numpy as np

vid = cv2.VideoCapture('avengersupdated.mp4')


def rescale_frame(frame, percent=75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

while True:
    ret, frame = vid.read()
    frame = rescale_frame(frame, percent=100)
    #gaussian = cv2.GaussianBlur(frame, (15, 15), 0)
    #median = cv2.medianBlur(frame, 1)
    #biletaral = cv2.bilateralFilter(frame, 2, 2, 5)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #averaging = cv2.blur(gray, (20, 20))
    #averaging2 = cv2.blur(averaging, (1, 1))
    denoise = cv2.fastNlMeansDenoisingColored(frame, None, 7, 15, 7, 21)
    #denoise2 = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    #ret, thresh = cv2.threshold(denoise, 80, 255, cv2.THRESH_BINARY_INV)
    #retval, threshold = cv2.threshold(gray, 88, 255, cv2.THRESH_BINARY)
    #kernel = np.array([[-1, -1, -1, -1, -1],
                        #[-1, 2, 2, 2, -1],
                        #[-1, 2, 8, 2, -1],
                        #[-1, 2, 2, 2, -1],
                        #[-1, -1, -1, -1, -1]]) / 4.0

    gray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    #denoise2 = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)
    #output_kernel = cv2.filter2D(denoise2, -1, kernel_sharpen)
    #cv2.imshow("kernel", output_kernel)
    #cv2.imshow("original", frame)
    #cv2.imshow('gaus', gaussian)
    #cv2.imshow('median', median)
    #cv2.imshow('bileteral', bileteral)
    cv2.imshow('gray', gray)
    #cv2.imshow('average', averaging)
    #cv2.imshow('average2', averaging2)
    cv2.imshow("denoise", denoise)
    #cv2.imshow("thresh", thresh)
    #cv2.imshow("threshold", threshold)

    c = cv2.waitKey(10) & 0xff
    if c == 27:
        break

frame.release()
cv2.destroyAllWindows()