import os #ipmort os module for interacting with the operating system
import numpy as np #import numpy for numerical operations and array handling
from sklearn.svm import SVC #import SVC (Support Vector Classifier) from scikit-learn for classification
from sklearn.model_selection import cross_val_score # used for cross validation of model. In other words, it helps measure how accurate your model is likely to be when used on data it hasn't seen before
import joblib #used for saving and loading python objects (like trained models)
from skimage.io import imread #reads images from files
from skimage.filters import threshold_otsu #used for automatic image thresholding

letters = [ #define the list of possible characters in license plate
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

def read_training_data(training_directory): #function for reading training data from a directory
    image_data = [] #Each of the 20x20 images of letters in the training set will be flattened into a 1D array and stored inside of this list
    target_data = [] #This list will store the actual characters that each image in the training data is supposed to represent.

    for each_letter in letters: #loop through each character in letters
        for each in range(10): #for each character loop through ten images (assumed to be named 0-9)
            image_path = os.path.join(training_directory, each_letter, each_letter + '_' + str(each) + '.jpg') #build the path to the image file
 
            img_details = imread(image_path, as_gray=True) #read image as a grayscale image

            binary_image = img_details < threshold_otsu(img_details) #convert the grayscale image to a binary image using Otsu's threshold
            # the @D array of each image is flattened because the machine learning
            # classifier requires that each sample is a 1D array
            # therefore the 20x20 image becomes 1*400
            # in machine learning terms that's 400 features with each pixel
            # representing a feature
            flat_bin_image = binary_image.reshape(-1) #flattens the 2D binary image into a 1D array (required for ML model)
            image_data.append(flat_bin_image) #add flattened image data to the list
            target_data.append(each_letter) #add images corresponding letter to the list

    return(np.array(image_data), np.array(target_data)) #converts list to numpy arrays and returns them
    
def cross_validation(model, num_of_fold, train_data, train_label):
    # this uses the concept of cross validation to measure the accuracy
    # of a model, the num_of_fold determines the type of validation
    # e.g if num_of_fold is 4, then we are performing a 4-fold cross validation
    # it will divide the dataset into 4 and use 1/4 of it for testing
    # and the remaining 3/4 for the training
    accuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold) #perform cross validation and get the accuracy scores for each fold. Returns a NumPy array of scores, one score for each fold.

    print("Cross Validation Result for", str(num_of_fold), "-fold")

    print(accuracy_result * 100)


current_dir = os.path.dirname(os.path.realpath(__file__)) #get the directory of the current script file

training_dataset_dir = os.path.join(current_dir, 'training_data') #build the path to the training dataset directory (assumed to be 'train' folder in current directory)

image_data, target_data = read_training_data(training_dataset_dir) #read the training data and labels from the dataset directory

# the kernel can be 'linear', 'poly', or 'rbf'
# the probability was set to True so as to show
# how sure the model is of it's prediction
svc_model = SVC(kernel='linear', probability=True) #creates a Support Vector Classifier (SVC) model from scikit-learn with a linear kernal and enables probability estimates

cross_validation(svc_model, 4, image_data, target_data) #perform 4-fold cross validation on the model using the training data

# let's train the model with all the input data
svc_model.fit(image_data, target_data) #this line trains the SVC using the training data. After this line, svc_model has learned to recognize the patterns in your training data and can be used to make predictions on new images

# we will use the joblib module to persist the model
# into files. This means that the next time we need to
# predict, we don't need to train the model again
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + '/svc.pkl')
#save_directory is being made equal to its absolute path. If it it doesn't exist, then make it. Then joblib.dump saves the trained svc model object to a file named svc.pkl in the specified directory. You can later load this model using joblib.load() when you want to make predictions without retraining.




#cross-validation splits your dataset into several parts (called "folds"). For each fold, it trains the model on the other folds and tests it on the current fold. This helps estimate how well your model will perform on unseen data.
#The cross_val_score function returns a numpy array containing the accuracy for each fold. For example, if you use 4 fold cross validation, it returns an array with 4 accuracy scores-one for each fold.

#os.path.join combines one or more string into a single file path, using the correct separator for your operating system (windows, mac/linux)
#os.path.join('folder', 'subfolder', 'file.txt')
# On Mac/Linux: 'folder/subfolder/file.txt'
# On Windows: 'folder\\subfoler\\file.txt'

#os.path.realpath(__file__)
#returns the absolute path to the current script file

#os.path.dirname()
#takes the absolute path of a file as an input and then it outputs the absolute path to the directory that the file is in


#there will be 340 entries in both image_data and target_data. The letters list has 34 characters, and for each character, there are 10 images. The first 10 entries in both image_data and target_data will correspond to the first chatacter in the letters list (which is '0')

