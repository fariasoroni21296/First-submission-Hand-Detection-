At file detect.py:

1. Line 7 - 13: importing necessary libraries torch (for pytorch), torch optimizer, torchvision for vision related data procesing, PILLOW library for making numpy array to Image. trackeback
2. Line 14, 15: defining the device. if NVIDIA GPU with CUDA available, pytorch will work in GPU if not then cpu
3. Line 28 - 47: detecting contours of written digit from a large numpy array and concatenating it.
                line 31: thresholding the numpy array
                line 33: finding the contours of the thresholded array
                line 34: defining the area
                line 35-35: taking the bounding rectangle from the contour array and apending it to areas array
                line 39-40: taking first element of an array
                line 42: sorting the areas array, the sent parameters are: function sortSecond, reverse sorting
                line 43-45: taking bounding x, y, width, and height and then drawing the rectangle in the corresponding line, concating the main array (gray = canvas) and putting it into a new array in the following line
                line 47: returning the concatenated array.

4. line 49-61: defining my sequential model; 
            from line 54 to 49 at first defined the linear model with 784 and 128 input and hidden size for the next layer.
            then passing the network to Rectified Linear Unit (ReLU) function to activate the nodes
            then again passing the activated nn to next Linear (aka Feed Forward neural network) with 128 x 64 input and output size.
            again relu in the next line
            then again Feed Forward (Linear) nn with 64 (input) and 10 (total number of digits) output.
            lastly, softmax function with logarithm to activate the column(hence dim = 1) of each digit.

5. Line 97 - 107:
            line 97: drawing the canvas with  numpy array and all the values are set to 0 with video shape.
            line 98: far_points to keep track of the hand index fingertip
            line 99 - 100: drawing indicator boolean variable and previous index tip tracker
            line 102-105: initializing the model, taking the model to either 'cpu' or 'gpu', loading the weights of the trained model in line 104 and 105.
            line 107: printing the model

6. line 123: keeping track of button press in key variable.
7. line 141, 143, 175: if key is equal character d => system will start drawing, if key is equal to c then cancel drawing, and if key is 's' then let the model decide which character it is
8. line 144: if c is pressed then drawing = False, canvas is again set to 0 and deleting the points of far_points array.
9. line 156-157 and 161-162: 
            if finger tip is fund then prev_index_finger_tip is stored from hte points_list's finger tip value's array.
            else if finger tip is not found then circle is drawin with previous index finger tip.

10. line 165-173: if drawing is True:
            line 165-166: if length of far_points array is greater than 100 delete the first value of the far_points array.
            line 167-168: if no fingertip is detected then the previous point is appended again.
11. line 171-173: each point before the last one on far_points a line in main frame and the canvas frame is drawn, the lines indicate the writing that I am trying to write on the screen in the air.

12. line 176-205: 
            line 176, 178drawing is turned down to false. canvas is flipeed for the model readability.
            line 181-184: resizing the image to 28x28 as the model receives this dimension, denoising the image by blurring with Gaussian distribution, showing the image and then sending the image array to create image.
            line 186-191: preproceesing the image with tensforms.Compose function: array_to tensor, normalizing.
            line 194: as the model is linear, it takes only one dimension of the image so image size is then 28*28 = 784 by 1.
            line 195-201: taking the model to no gradiant calculation stage. predict the model with log_softmax funtion (last layer of the model, remember?), printing the log_prediction, using torch.exp to increase the values exponentially and taking out the real probabilities, converting the probabilities to numpy array by taking the array from cpu to gpu and then converting the array to list and taking the top (zero) prediction.

13. line 209: showing the canvas image.


