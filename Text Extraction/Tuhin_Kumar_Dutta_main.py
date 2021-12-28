# Asking for pdf path as input
path = input('Enter the path of pdf: ')


# IMPORTS
from pylab import rcParams
# Setting the image dimensions
rcParams['figure.figsize'] = 10,20
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import easyocr
from nltk.tokenize import sent_tokenize
from pdf2image import convert_from_path


# reader object from easyocr library
reader = easyocr.Reader(['en'])


# Function to plot bounding boxes around texts
def plot(image):
    # Reading pdf from the provided path
    output = reader.readtext(image)
    # Creating a dataframe of the coordinates
    cordinates = pd.DataFrame(columns=['x_min','y_min','x_max','y_max'])
    for index,i in enumerate(output):
        # Selecting the coordinates element from the output
        cord = i[0]
        # Coordinates from the 'cord'
        x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
        x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
        # Imputing the dataframe with the coordinates values from each bounding box
        cordinates.loc[index] = [x_min,y_min,x_max,y_max]
    # Using the opencv library to read the image
    pic = cv2.imread(image)
    # Drawing bounding box for each coordinates in the dataframe
    for i in cordinates.index:
        cv2.rectangle(pic,(cordinates.loc[i,'x_min'],cordinates.loc[i,'y_min']),(cordinates.loc[i,'x_max'],cordinates.loc[i,'y_max']),(0,0,255),2)
    # Converting BGR to RGB format, so that matplotlib can be used to save the image
    plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    # Saving the image with texts within bounding boxes, with the same name as that of the passed image
    plt.savefig(image.split('.')[0]+'_with_bounding_boxes.png')
    # Joining all the texts in the output texts and performing sentence tokenization from nltk
    texts = ' '.join([i[1] for i in output])
    return sent_tokenize(texts)


# Function to convert provided pdf from path to image
def pdf_parsing(path):   
    # Converting pdf pages to images and assigning the images to the variable 
    images = convert_from_path(path)
    # Initiating a blank list
    texts = []
    for i in range(len(images)):
        # Saving each image from 'images'
        images[i].save('page'+ str(i+1) +'.jpg', 'JPEG')
        # Implementing the 'plot' function for each images
        text = plot('page'+ str(i+1) +'.jpg')
        # Appending the texts from each pdf page to the blank list 'texts'
        texts.append(text)        
    return texts


# Implementing 'pdf_parsing' function on the input 'path'
texts = pdf_parsing(path)


# A text file is generated
file = open('extracted_texts.txt', 'w')
# The extracted text files are saved in the generated text file
for i in range(len(texts)):
    file.write(f'Texts in page{i+1}:-\n\n')
    for j in texts[i]:
        file.write(f'{j}\n\n')
    file.write('\n\n\n')
file.close()
