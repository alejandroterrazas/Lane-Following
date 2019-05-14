import cv2
import numpy as np

                         
def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 1)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary_output = np.zeros_like(gray)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    sm = np.sqrt(np.power(sx,2) + np.power(sy,2))
    
    scaled = sm/np.max(sm).astype(np.uint8)
    #plt.imshow(scaled)
    binary_output[np.where((scaled>mag_thresh[0]) &
                           (scaled<mag_thresh[1]))] = 1
                           
    
    return binary_output

def abs_sobel_thresh(gray, orient='x', thresh_min=0, thresh_max=1):
    # Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def canny_edge(gray, kernel = 3, sigma=.33):
    #blurred = cv2.GaussianBlur(gray, ksize=(kernel, kernel),
    v = np.median(gray)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    return edges//np.max(edges).astype('uint8')    


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
   # print(np.unique(absgraddir[np.where(absgraddir>0)]))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output.astype('uint8')

def color_mask(img):
  '''extracts white and yellow pixkes from tricolor image'''
  mask = np.zeros([img.shape[0], img.shape[1]])
  mask[np.where((img[:,:,0]>200) & (img[:,:,1]>200) & (img[:,:,2]>200))] = 255
  
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  
  mask[np.where((hsv[:,:,0]>20) & (hsv[:,:,0]<110) &
                 (hsv[:,:,1]>50) & (hsv[:,:,2]>50))] = 255
  return mask


def white_mask(img):
  '''extracts white pixels tricolor image'''
  mask = np.zeros([img.shape[0], img.shape[1]])
  mask[np.where((img[:,:,0]>200) & (img[:,:,1]>200) & (img[:,:,2]>200))] = 255
  return mask
 
def yellow_mask(img):
  '''extracts in HLS pixels tricolor image'''
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = np.zeros([img.shape[0], img.shape[1]])
  mask[np.where((hsv[:,:,0]>20) & (hsv[:,:,0]<110) &
                 (hsv[:,:,1]>50) & (hsv[:,:,2]>50))] = 255
  return mask

                         
def morph_ops(binary):

     dilate = cv2.dilate(binary,kernel,iterations = 3)       
     erosion = cv2.erode(binary,kernel,iterations = 1)
     return dilate    