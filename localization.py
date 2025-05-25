from skimage.io import imread #import function to read images from skimage
from skimage.filters import threshold_otsu #import the Otsu thresholding function from skimage. This function calculates an optimal threshold value for creating a grayscale image using Otsu's method. This value separates the pixel intensities into two classes (foreground and background).
import matplotlib.pyplot as plt # import matplotlib for plotting images

car_image = imread("car.jpg", as_gray=True) #reads the image "car.jpg" in a grayscale mode as a 2D array. Each row and column in the array corresponds to a pixel in the image. Each entry in the array is the intensity or brightness of the pixel. This value for the intensity is between 0 (black) and 1 (white) in skimage.
#it should be a 2 dimensional array
print(car_image.shape)#prints the length and width of the image in pixels

#the next line is optional. A gray scale pixel in skimage ranges between 0 and 1. Multiplying it with 255 will make it easier to relate to.

gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)#creates a single figure with a grid of subplots arranged in 1 row and 2 columns (so one left one right). The subplot on the left is ax1 and the subplot on the right is ax2.
ax1.imshow(gray_car_image, cmap="gray")#Displays the grayscale image in the first subplot
threshold_value = threshold_otsu(gray_car_image)#Calculates the threshold value for the grayscale image to distinguish between foreground and background pixels.
binary_car_image = gray_car_image > threshold_value #Create a binary image by thresholding: pixels above threshold are True (white) and others are False (black). binary_car_image is a 2D array of booleans.
ax2.imshow(binary_car_image, cmap="gray") #Display the binary image in the second subplot
plt.show() #Show both images


#This file is made to identify the license plate's position in the image. In order to do this we need to read the image and convert it to grayscale image where each pixel is between a value of 0 & 255 intensity (brightness). Then we convert it to a binary image in which a pixel is either black or white.