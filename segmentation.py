import numpy as np # import numpy for numerical operations
from skimage.transform import resize # import resize function to resize images
from skimage import measure # import measure module for image region analysis
from skimage.measure import regionprops # import regionproprs to get properties of labeled regions
import matplotlib.patches as patches # import patches for drawing rectangles on images
import matplotlib.pyplot as plt # import pyplot for plotting images
import cca2 #import cca2 module (contains plate_like_objects)

# on the image I'm using, the headlamps were categorized as a license plate
# because their shapes were similar
# for now I'll just use the plate_like_objects[2] since I know that's the
# license plate. We'll fix this later

# the invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(cca2.plate_like_objects[2]) #invert the image so the black pixels become white and vice versa

labelled_plate = measure.label(license_plate) #label connected regions in the binary license plate image

fig, ax1 = plt.subplots(1) #create a matplotlib figure and axis for displaying the image
ax1.imshow(license_plate, cmap="gray") #show the inverted license plate image in grayscale


#Define character size constraints based on license plate dimensions: Height: 35% to 60% of plate height. Width: 5% to 15% of plate width. This will eliminate some.

character_dimensions = (0.35*license_plate.shape[0], 0.60*license_plate.shape[0], 0.05*license_plate.shape[1], 0.15*license_plate.shape[1])
min_height, max_height, min_width, max_width = character_dimensions

characters = [] #will store character images. Each image is a 2D array
counter = 0 #counts the number of characters
column_list = [] #stores the x-coordinates of detected characters for sorting
for regions in regionprops(labelled_plate):#iterate through each detected region in labeled plate image
    y0, x0, y1, x1, = regions.bbox
    region_height = y1 - y0
    region_width = x1 - x0 #get region_height and region_width

    if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
        roi = license_plate[y0:y1, x0:x1] #extract region of interest (ROI) corresponding to that character. roi is a 2D array with only the pixels inside the bounding box for the detected character region

        # draw a red bordered rectangle over the character
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor = "red", linewidth = 2, fill = False) #draw a red rectangle around the detected character on the plot

        ax1.add_patch(rect_border)

        # resize the characters to 20x20 and then append each character into the characters list
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

plt.show()


#This part of the program actually detects the individual characters inside of the license plate and shows a plot of the license plate, with red rectangles around each of the characters.

#the measure.label function scans the input image (which should be binary: foreground pixels are 1 or True or white, and background pixels are 0 or False or black). A binary image is a 2D array where each value is either a 0 or 1, for background or foreground, black or white. The output of this function takes that binary image and labeles regions with a lot of connected white pixels. It outputs a 2D array where the first grouped region is all assigned a value of 1, the second grouped region is all assigned a value of 2, the third grouped region is all assigned a value of 3, etc. And the background pixels are still 0.
#for example
# Input (binary image)
#   [[0, 1, 1, 0, 0],
#   [0, 1, 1, 0, 1],
#   [0, 0, 0, 0, 1]]

# Output (labeled image)
#   [[0, 1, 1, 0, 0],
#   [0, 1, 1, 0, 2],
#   [0, 0, 0, 0, 2]]

#bounding box
#y0, x0, y1, x1 = regions.bbox
#y0: Top row index of the bounding box (start of the region, vertically)
#x0: Left column index of the bounding box (start of the region, horizontally)
#y1: Bottom row index (end of the region, vertically; exclusive)
#x1: Right column index (end of the region, horizontally; exclusive)
