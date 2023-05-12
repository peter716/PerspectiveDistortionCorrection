# -*- coding: utf-8 -*-
#!/Users/Hp/AppData/Local/Programs/Python/Python311

"""
Created on Fri Nov 11 21:03:20 2022

@author: oadedeji
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.linalg import inv


#To run, use python filename.py

def generate_H_inverse(processed_cord, distorted_cord):
    """
    

    Parameters
    ----------
    processed_cord : (row,col)
    distorted_cord : (row,col)

    Returns
    -------
    h_inv : 3 by 3 matrix

    """
    # Extract the X matrix
    X = np.array([[distorted_cord[0],distorted_cord[1],1,0,0,0, -processed_cord[0]*distorted_cord[0], -processed_cord[0]*distorted_cord[1]],[0,0,0,distorted_cord[0],distorted_cord[1],1,-processed_cord[1]*distorted_cord[0],-processed_cord[1]*distorted_cord[1]],[distorted_cord[2],distorted_cord[3],1,0,0,0, -processed_cord[2]*distorted_cord[2], -processed_cord[2]*distorted_cord[3]],[0,0,0,distorted_cord[2],distorted_cord[3],1,-processed_cord[3]*distorted_cord[2],-processed_cord[3]*distorted_cord[3]],[distorted_cord[4],distorted_cord[5],1,0,0,0, -processed_cord[4]*distorted_cord[4], -processed_cord[4]*distorted_cord[5]],[0,0,0,distorted_cord[4],distorted_cord[5],1,-processed_cord[5]*distorted_cord[4],-processed_cord[5]*distorted_cord[5]],[distorted_cord[6],distorted_cord[7],1,0,0,0, -processed_cord[6]*distorted_cord[6], -processed_cord[6]*distorted_cord[7]],[0,0,0,distorted_cord[6],distorted_cord[7],1,-processed_cord[7]*distorted_cord[6],-processed_cord[7]*distorted_cord[7]]])
    
    #Get the inverse of X and comfirm that the product of inverse and X gives identity matrix
    distorted_inv = inv(X)
    np.allclose(np.dot(X, distorted_inv), np.eye(8))
    
    #Get the H matrix
    H = np.dot(distorted_inv, processed_cord)
    
    #Convert the array to a list
    h_reshaped = list(H)
    
    #Append 1 to the list so as to have 9 items(to be able to reshape to 3 by 3)
    h_reshaped.append(1)
    
    #Convert back to an array and reshape to 3 by 3
    h_reshaped = np.array(h_reshaped)
    h_reshaped = h_reshaped.reshape(3,3)
    
    # Get the inverse of the resultant H matrix and return it
    h_inv = inv(h_reshaped)
    
    return h_inv


    
    
    
def perspective_distortion(image_array, processed_array,h_inv):
    """
    

    Parameters
    ----------
    image_array : Image array
    processed_array : Processed array
    h_inv : H inverse

    Returns
    -------
    processed_array : resultant processed array

    """
    
    # Loop through all the pixels in the processed array
    #For every i and j in the processed array, get the corresponding c and d in the distortion array
    #Place the RGB value at that c,d pixel value in the i,j position
    #Return the processed array
    
    for i in range(processed_array.shape[0]):
        for j in range(processed_array.shape[1]):
            a_b = np.array([i,j,1])
            c_d = np.dot(h_inv,a_b)
            c = int(c_d[0]/c_d[2])
            d = int(c_d[1]/c_d[2])
            
            if (c >= 0 and c < image_array.shape[0]) and (d >= 0 and d < image_array.shape[1]):
                processed_array[i,j] = image_array[c,d]
                
    return processed_array
               
#Load the image, extract the array, and set the processed array as zero    
my_image1 = Image.open("PC_test_1.jpg")
image_array_1 = np.asarray(my_image1, np.float64)
processed_array_1 = np.where(image_array_1 <= 256, 0,0)

my_image2 = Image.open("PC_test_2.jpg")
image_array_2 = np.asarray(my_image2, np.float64)
processed_array_2 = np.where(image_array_2 <= 256, 0,0)

#Get the coordinates distorted = [c1,d1,c2,d2,c3,d3....], likewise for processed (a,b)
processed_cord_1 = [433,137,345,137,345,477,433,477]
distorted_cord_1 = [433,137,345,179,345,425,431,477]

processed_cord_2 = [540,473,324,473,324,621,540,621]
distorted_cord_2 = [500,469,343,473,324,621,540,617]

#Get the H inverse
h_inv_1 = generate_H_inverse(processed_cord_1,distorted_cord_1)
h_inv_2 = generate_H_inverse(processed_cord_2,distorted_cord_2)

#Perform perspective distortion correction
processed_array_1 = perspective_distortion(image_array_1, processed_array_1,h_inv_1)
processed_array_2 = perspective_distortion(image_array_2, processed_array_2,h_inv_2)

#Save the processed image in that directory
Image.fromarray(processed_array_1.astype(np.uint8)).save("processed_test_1.jpg")
Image.fromarray(processed_array_2.astype(np.uint8)).save("processed_test_2.jpg")

#Show that operation has been completed
print("Done")
