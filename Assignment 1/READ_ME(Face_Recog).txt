------------------------------------Face Recognition--------------------------------------

-Place all the python files along with the database in the same folder

-Run the file 'Face_Recog.py'

-The dataset used for this assignment is ORL Face Database, the link of which is mentioned in the report. It consists of 40 subjects with 10 images each. Each image is of the size 112x92x3.

-The program takes 8 images each of the 40 subjects for training(320) and the remaining 2 images of each subject is used for testing(80).

-Both the training and testing codes are in the same file(code for training followed by code for testing).
NOTE:If the code is run in Spyder then the training and testing are separated into cells and can be run separately

-The program prints the % of accuracy of prediction in the end

-The matrix 'pred' shows what the prediction is. The rows of the column represent the subject number and the column represent the test image number and values represent the prediction(predicted subject number)

-To display an image(average image, eigenfaces), the function disp_img() can be used which has the arguments(image, row, column, depth)
