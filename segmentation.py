import numpy as np # import numpy for numerical operations
from skimage.transform import resize # import resize function to resize images
from skimage import measure # import measure module for image region analysis
from skimage.measure import regionprops # import regionproprs to get properties of labeled regions
import matplotlib.patches as patches # import patches for drawing rectangles on images
import matplotlib.pyplot as plt # import pyplot for plotting images
from skimage.filters import * #import the Otsu thresholding function from skimage. This function calculates an optimal threshold value for creating a grayscale image using Otsu's method. This value separates the pixel intensities into two classes (foreground and background).
import cca2 #import cca2 module (contains plate_like_objects)


# This is another method for determining which index in plate_like_objects is the real license plate. In the code below I detect characters in each plate like object and I count them. Then I have a for loop that states that the plate_like_object with the highest number of detected characters, will have its index saved to licensePlateIndex.
def countChars(PLO):
    counter = 0
    invPLO = np.invert(PLO)
    labeledPLO = measure.label(invPLO)

    #Define character size constraints based on license plate dimensions: Height: 35% to 60% of plate height. Width: 5% to 15% of plate width. This will eliminate some.
    character_dimensions = (0.35*invPLO.shape[0], 0.60*invPLO.shape[0], 0.05*invPLO.shape[1], 0.15*invPLO.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    for regions in regionprops(labeledPLO):#iterate through each detected region in labeled plate image
        y0, x0, y1, x1, = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0 #get region_height and region_width

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            counter += 1
    return counter

# print("chars in plo index 0", countChars(cca2.plate_like_objects[0]))
# print("chars in plo index 1", countChars(cca2.plate_like_objects[1]))
# print("chars in plo index 2", countChars(cca2.plate_like_objects[2]))

licensePlateIndex = 0
if len(cca2.plate_like_objects) > 1:
    maxCount = countChars(cca2.plate_like_objects[0]) #maxCount variable to find plate_like_object with the most detected characters
    maxIndex = 0 #I want the index of the plate_like_object that has the most detected characters in it
    for i in range(len(cca2.plate_like_objects)):

        if countChars(cca2.plate_like_objects[i]) > maxCount:
            maxCount = countChars(cca2.plate_like_objects[i])
            maxIndex = i

    licensePlateIndex = maxIndex





#now that we know which part of the picture is the license plate I want to do a threshold on just the license plate. When I thresholded the whole picture, it might have cut out some pixels in the license plate itself. If I do another threshold on just the license plate to create a binary picture, then it will be more accuracte.

gray_license_plate = cca2.gray_plate_like_objects[licensePlateIndex]
# threshold_value = int(input("please enter a threshold"))
# threshold_value = 120#Calculates the threshold value for the grayscale image to distinguish between foreground and background pixels.
threshold_value = threshold_otsu(gray_license_plate)
binary_license_plate = gray_license_plate > threshold_value #Create 

print("Threshold Value:", threshold_value)

plt.figure()
plt.imshow(binary_license_plate, cmap='gray')
plt.title("Binary License Plate")
plt.axis('off')
plt.show()



#Here I want to detect characters in the license plate again. It may not detect all of them because there could be a shadow or a design in the license plate that connects to one of the letters, making that letter not be detected. So what I will do is for the letters it does detect, I will take the lowest point that any detected character reaches, and I will basically cut off all of the content below that since all of the letters are supposed to be inline. This way I can detect characters again, and every character will correctly be detected.


def cutOffContent(license_plate):
    counter = 0
    highestPoints = []
    lowestPoints = []
    inv_license_plate = np.invert(license_plate)
    labeled_license_plate = measure.label(inv_license_plate)

    #Define character size constraints based on license plate dimensions: Height: 35% to 60% of plate height. Width: 5% to 15% of plate width. This will eliminate some.
    character_dimensions = (0.35*inv_license_plate.shape[0], 0.60*inv_license_plate.shape[0], 0.05*inv_license_plate.shape[1], 0.15*inv_license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    for regions in regionprops(labeled_license_plate):#iterate through each detected region in labeled plate image
        y0, x0, y1, x1, = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0 #get region_height and region_width

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            highestPoints.append(y0)
            lowestPoints.append(y1)
            counter += 1

    if counter > 0:
        highestPoint = min(highestPoints)
        lowestPoint = max(lowestPoints)
        license_plate[:highestPoint, :] = True
        license_plate[lowestPoint:, :] = True

cutOffContent(binary_license_plate)
#binary_license_plate[-10:, :] = True

plt.figure()
plt.imshow(binary_license_plate, cmap='gray')
plt.title("Binary License Plate")
plt.axis('off')
plt.show()







# the invert was done so as to convert the black pixel to white pixel and vice versa
license_plate = np.invert(binary_license_plate) #invert the license plate image so the black pixels become white and vice versa

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
        roi = license_plate[y0:y1, x0:x1] #extract region of interest (ROI) corresponding to that character. roi is a 2D array with only the pixels inside the bounding box for the detected character region. Its like a cropped image of just the character.

        counter += 1

        # draw a red bordered rectangle over the character
        rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor = "red", linewidth = 2, fill = False) #creates red rectangle around detected character

        ax1.add_patch(rect_border)#adds the red rectangle to the subplot

        # resize the characters to 20x20 and then append each character into the characters list
        resized_char = resize(roi, (20, 20))
        characters.append(resized_char)

        # this is just to keep track of the arrangement of the characters
        column_list.append(x0)

plt.show()#show the plot
print("Characters Detected:", counter)

#shows the resized characters -delete later
if characters:
    # Pair each character with its x-coordinate, then sort by x-coordinate
    sorted_chars = [char for _, char in sorted(zip(column_list, characters))]
    plt.figure(figsize=(10, 2))
    for i, char_img in enumerate(sorted_chars):
        plt.subplot(1, len(sorted_chars), i + 1)
        plt.imshow(char_img, cmap='gray')
        plt.axis('off')
    plt.show()
else:
    print("No characters detected.")





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


#The measure.label function will label any group of at least one connected foreground pixel (value 1 or True) as a region. There is no minimum size by default so even a single pixel will be labeled as its own region. If you want to filter out small regions, you nmeed to do that yourself (as you do with if region.area < 50:    continue)


#bounding box
#y0, x0, y1, x1 = regions.bbox
#y0: Top row index of the bounding box (start of the region, vertically)
#x0: Left column index of the bounding box (start of the region, horizontally)
#y1: Bottom row index (end of the region, vertically; exclusive)
#x1: Right column index (end of the region, horizontally; exclusive)