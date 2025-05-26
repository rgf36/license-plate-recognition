from skimage import measure #Import the measure module from skimage, which provides functions for image analysis
from skimage.measure import regionprops #Import the regionprops function from skimage.measure to get properties of labeled regions. Properties could be area, bounding box (smallest rectangle to contain region), centroid (center of region), eccentricity (shape) or orientation.
import matplotlib.pyplot as plt #Import to plot images
import matplotlib.patches as patches #Import the patches module for drawing shapes (like rectangles) on images
import localization 

#This gets all the connected regions and groups them together
label_image = measure.label(localization.binary_car_image) #label the connected regions in binary car image
fig, (ax1) = plt.subplots(1) #create a figure and a subplot (ax1) for displaying the image. The 1 next to subplots means there is one row and one column of one subplot in the figure.
ax1.imshow(localization.gray_car_image, cmap="gray"); #Display the grayscale car image on the subplot using a gray colormap

#regionprops creates a list of properties of all the labelled regions. region is an object -specifically an instance of the RegionProperties class. You access the properties as attributes of the object (.area, .bbox, .centroid etc)
for region in regionprops(label_image):#iterate over each connected region found in the labeled image
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue

    #the bounding box coordinates for the region (min row, min col, max row, max col)
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)#creates a patch or a shape in matplotlib of a red rectangle around the detected region in the image. rectBorder is an object of the Rectangle class in matplotlib.
    ax1.add_patch(rectBorder)#adds the rectangle over the detected region


plt.show()


#Here we need to identify all the connected regions in the image, using a concept called connected component analysis (CCA). CCA helps us group and label connected regions on the foreground. A pixel is deemed to be connected to another if they both have the same value and are adjacent to each other.


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


#The measure.label function will label any group of at least one connected foreground pixel (value 1 or True) as a region. There is no minimum size by default so even a single pixel will be labeled as its own region. If you want to filter out small regions, you nmeed to do that yourself (as you do with if region.area < 50:    continue)


#bounding box
#y0, x0, y1, x1 = regions.bbox
#y0: Top row index of the bounding box (start of the region, vertically)
#x0: Left column index of the bounding box (start of the region, horizontally)
#y1: Bottom row index (end of the region, vertically; exclusive)
#x1: Right column index (end of the region, horizontally; exclusive)