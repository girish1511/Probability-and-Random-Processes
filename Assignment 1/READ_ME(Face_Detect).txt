------------------------------------Face Detection----------------------------------------

-Place all the python files along with the database in the same folder

-Run the file 'Face_Detect.py'

-The dataset used for this assignment is ORL Face Database, the link of which is mentioned in the report. It consists of 40 subjects with 10 images each.  Each image is of the size 112x92x3.

-The program takes 10 images each of the 35 subjects(350 images) for training and 10 images each of remaining 5 subjects is used for testing(50).

-Both the training and testing codes are in the same file(code for training followed by code for testing). 
NOTE:If the code is run in Spyder then the training and testing are separated into cells and can be run separately

-The program prints the plot of accuracy vs threshold and error vs accuracy with threshold varying from (0,7500)

-The program then prints the % of accuracy of prediction in the end for a particular value of threshold(6000)

-The matrix 'pred' shows what the prediction is. The rows of the column represent the subject number and the column represent the test image number and values represent the prediction(True corresponds to face detected and False corresponds to face not detected).

-To display an image(average image, eigenfaces), the function disp_img() can be used which has the arguments(image, row, column, depth)
