from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import localization

# this gets all the connected regions and groups them together
label_image = measure.label(localization.binary_car_image) #label the connected regions in binary 

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])#making a tuple called plate_dimensions with min_height, max_height, min_width, max_width. label_image.shape is a tuple with the height and width of the image. So, label_image[0] is the height in pixels, and label_image[1] is the width in pixels.
min_height, max_height, min_width, max_width = plate_dimensions#"unpacking" the tuple into the variables
plate_objects_cordinates = []
plate_like_objects = []
fig, (ax1) = plt.subplots(1) #create a figure and a subplot (ax1) for displaying the image. The 1 next to subplots means there is one row and one column of one subplot in the figure.
ax1.imshow(localization.gray_car_image, cmap="gray"); #Display the grayscale car image on the subplot using a gray colormap

#regionprops creates a list of properties of all the labelled regions. region is an object -specifically an instance of the RegionProperties class. You access the properties as attributes of the object (.area, .bbox, .centroid etc)
for region in regionprops(label_image):
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox
    region_height = max_row - min_row
    region_width = max_col - min_col
    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        plate_like_objects.append(localization.binary_car_image[min_row:max_row, min_col:max_col]) #localization.binary_car_image is the black and white image of the car. This uses slicing to extract a rectangular region from the image. These "cropped" images are stored in the plate_like_objects list.

        plate_objects_cordinates.append((min_row, min_col, max_row, max_col)) #for each plate_like_object, it stores the bounding box coordinates of it. min_row is the topmost row of the region. min_col is the leftmost column of the region. max_row is the bottommost row of the region. max_col is the rightmost row of the region. In other words it stores the pixel coordinates for the highest, lowest, most left and most right points.

        rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)#creates a red rectangle surrounding the region
        ax1.add_patch(rectBorder)#adds the rectangle to the subplot


plt.show()




#From the resulting image in cca.py, we can see that other regions that do not contain the license plate are also mapped. In order to eliminate these, we will use some characteristics of a typical license plate to remove them:
#1. they are rectangular in shape
#2. The width is more than the height
#3. The proportion of the width of the license plate region to the full image ranges between 15% and 40%
#4. The proportion of the height of the license plate region to the full image is between 8% & 20%






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