
import os
import numpy as np
import matplotlib.image as mpimg
 

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
   

    return np.asarray(imgs)

def extract_test_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    imgs = []
    for i in range(1, num_images+1):
        folder_id = "test_%d/" % i
        imageid = "test_%d" % i
        image_filename = filename + folder_id + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
   

    return np.asarray(imgs)



#helper functions to convert images to grayscale
# see : https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale

def grayScale(imgs):

    nbr_img = imgs.shape[0]

    gray = []

    for i in range(nbr_img):

        gray.append(grayImage(imgs[i]))
      
    return np.asarray(gray)


def grayImage(img):

    gray_img = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):

      for j in range(img.shape[1]):

        pixel = img[i,j] / 255.0

        gray_img[i,j] = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

    return gray_img


#split data into testing and training set given a ratio
def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
 
    # split the data based on the given ratio

    training_nbr = int(x.shape[0] * ratio)
    indexes = np.random.choice(x.shape[0],training_nbr, replace=False)
    
    x_train = x[indexes]
    y_train = y[indexes]
    x_test = np.delete(x, indexes, axis = 0)
    y_test = np.delete(y, indexes, axis = 0)
    
    
    return x_train, y_train, x_test, y_test

