import numpy as np
import tensorflow as tf
from dataset import *
import os
import argparse
import cv2
from glob import glob
from keras.preprocessing import image
import random
from cozmo.util import degrees, distance_mm ,speed_mmps,Angle
import cozmo
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train CNN to classify images into N number of Classes.')
parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu
datasetTrain = load_cached(cache_path="../cache files/Train_cache.pkl",in_dir="../Dataset/Train")

num_classes_Train = datasetTrain.num_classes
datasetValidation = load_cached(cache_path="../cache files/Validation_cache.pkl",in_dir="../Dataset/Test")

num_classes_Validation = datasetValidation.num_classes
image_paths_train, cls_train, labels_train = datasetTrain.get_training_set()
image_paths_test, cls_test, labels_test = datasetValidation.get_training_set()

training_iters = 300
learning_rate = 0.001
batch_size = 1
n_input_W = 80
n_input_H = 80
n_classes = 4

#both placeholders are of type float
x = tf.placeholder("float", [None, n_input_H,n_input_W,3])

y = tf.placeholder("float", [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.layers.max_pooling2d(x, pool_size=[k, k], strides=[k, k, ], padding='SAME')

weights = {
    'wc1': tf.get_variable('W0', shape=(5,5,3,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wc3': tf.get_variable('W2', shape=(3,3,64,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wcAli': tf.get_variable('ali', shape=(5,5,64,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd1': tf.get_variable('crowd1', shape=(5,5,128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd2': tf.get_variable('crowd2', shape=(5,5,128,128), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd3': tf.get_variable('crowd3', shape=(5,5,128,512), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd4': tf.get_variable('crowd4', shape=(5, 5, 512, 512), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd5': tf.get_variable('crowd5', shape=(5, 5, 512, 256), initializer=tf.contrib.layers.xavier_initializer()),
    'crowd6': tf.get_variable('crowd6', shape=(5,5,256,64), initializer=tf.contrib.layers.xavier_initializer()),
    'wd1': tf.get_variable('W3', shape=(64,32), initializer=tf.contrib.layers.xavier_initializer()),
    'wd2': tf.get_variable('W4', shape=(64,32), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('W6', shape=(32,n_classes), initializer=tf.contrib.layers.xavier_initializer()),
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'BAli': tf.get_variable('BAli1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd1': tf.get_variable('Bcrowd1', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd2': tf.get_variable('Bcrowd2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd3': tf.get_variable('Bcrowd3', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd4': tf.get_variable('Bcrowd4', shape=(512), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd5': tf.get_variable('Bcrowd5', shape=(256), initializer=tf.contrib.layers.xavier_initializer()),
    'Bcrowd6': tf.get_variable('Bcrowd6', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bd2': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B5', shape=(4), initializer=tf.contrib.layers.xavier_initializer()),
}




def conv_net(x, weights, biases):

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    #conv1 = conv2d(conv1, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    conv3A = conv2d(conv3, weights['wcAli'], biases['BAli'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3A, k=2)

    conv4 = conv2d(conv3, weights['crowd1'], biases['Bcrowd1'])
    conv4A = conv2d(conv4, weights['crowd2'], biases['Bcrowd2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv4 = maxpool2d(conv4A, k=2)

    conv5 = conv2d(conv4, weights['crowd2'], biases['Bcrowd2'])
    conv5A = conv2d(conv5, weights['crowd3'], biases['Bcrowd3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv5 = maxpool2d(conv5A, k=2)


    conv6 = conv2d(conv5, weights['crowd4'], biases['Bcrowd4'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv6 = maxpool2d(conv6, k=2)

    conv7 = conv2d(conv6, weights['crowd5'], biases['Bcrowd5'])
    conv8 = conv2d(conv7, weights['crowd6'], biases['Bcrowd6'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv8 = maxpool2d(conv8, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    dropout1=tf.layers.dropout(conv8, rate=0.5)
    fc1 = tf.reshape(dropout1, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc3 = tf.nn.relu(fc2)
    # finally we multiply the fully connected layer with the weights and add a bias term.
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out


pred = conv_net(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


#Here you check whether the index of the maximum value of the predicted image is equal to the actual labelled image.
# and both will be a column vector.
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

#calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initializing the variables
init = tf.global_variables_initializer()
test_X = load_images(image_paths_test)

test_Y=labels_test
saver = tf.train.Saver()

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
total_iterations=0

if args.command == "train":
    print('\nTraining the Model...')
    with tf.Session() as sess:
        sess.run(init)
        for i in range(training_iters):
            j=0
            print("loading")
            print("iteration NO#")
            print(i)
            test_accuracy_count = 0
            avg_maker = 0
            for batch in range(len(image_paths_train)//batch_size):
                 total_iterations +=1
                 batch_paths,batch_y=random_batch(image_paths_train, batch_size, labels_train)
                 #print(batch_paths)
                 batch_x = load_images(batch_paths)
                 opt = sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                 if (total_iterations % 5 == 0) or (i == (j - 1)):
                     # Calculate the accuracy on the training-batch.
                     loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})

                     print("Batch " + str(j) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= "+ \
                      "{:.5f}".format(acc))
                 j=j+1
            print("Iter " + str(i) + ", Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            print("Optimization Finished!")
            #trying to load as batch data
            for batch in range(len(image_paths_test) // batch_size):
                    print(image_paths_test[batch * batch_size:min((batch + 1) * batch_size, len(image_paths_test))])
                    batch_test_x = load_images( image_paths_test[batch * batch_size:min((batch + 1) * batch_size, len(image_paths_test))])
                    batch_test_y = test_Y[batch * batch_size:min((batch + 1) * batch_size, len(test_Y))]
                    test_acc,valid_loss = sess.run([accuracy,cost], feed_dict={x: batch_test_x , y:batch_test_y})
                    train_loss.append(loss)
                    test_loss.append(valid_loss)
                    train_accuracy.append(acc)
                    test_accuracy.append(test_acc)
                    test_accuracy_count=test_accuracy_count+test_acc
                    avg_maker=avg_maker+1
                    print("Testing Accuracy:","{:.5f}".format(test_acc))
            avg_test_accuracy=test_accuracy_count/avg_maker
            print("Testing Accuracy on complete Test Data:", "{:.5f}".format(avg_test_accuracy))
        saver.save(sess,'Models/modle.ckpt')
        print("Model Saved Successfully:")
# elif (args.command=="test"):
#      with tf.Session() as sees:
#          saver.restore(sees, "Models/modle.ckpt")
#          print("Model Loaded Successfully:")
#          for i in range(10):
#              img=cv2.imread('FinalTesting/clastest('+str(i+1)+').png')
#              cv2.imshow('image',img)
#              cv2.waitKey(1)
#              my_np_array = img.reshape(1, 80, 80, 3)
#              output=tf.argmax(pred, 1)
#              className=sees.run(output, feed_dict={x: my_np_array})
#           print(className)

elif (args.command == "final"):
     dest_array=[]
     with tf.Session() as sees:
        saver.restore(sees, "Models/modle.ckpt")
        print("Model Loaded Successfully:")
        img_mask = 'FinalTesting/*.png'
        img_names = glob(img_mask)
        #for every images one by one print the class name
        for fn in img_names:
            img = cv2.imread(fn)
            print('processing %s...' % fn, )
            my_np_array = img.reshape(1, 80, 80, 3)
            output=tf.argmax(pred, 1)
            className=sees.run(output, feed_dict={x: my_np_array})
            print(className)

            dest_array.append(className)
        # print(dest_array)
        np_arr=np.array(dest_array)

        # print(np_arr)














        def cozmo_program(robot: cozmo.robot.Robot):
            #for converting the values according to the south,north,east,west
            for item in np_arr:
                result = np.where(np_arr == 0)
                #for down
                np_arr[result] = 270
                result2 = np.where(np_arr == 1)
                #for left
                np_arr[result2] = 180
                result3 = np.where(np_arr == 2)
                #for right
                np_arr[result3] = 360
                result4 = np.where(np_arr == 3)
                #for up
                np_arr[result4] = 90

            # print(np_arr)
            #starting position of the robot
            current = 270

            my_dict = {"North": "90", "South": "270", "East": "360", "West": "180"}
            print(my_dict)
            x=0
            y=0
            #according to the 2D array
            for item in range(len(np_arr)):
                #if it is the first element
                if item==0:
                    #Finding the Conditions with respect to the every point
                    find4conditions(0,0)
                    #Finding the next state of the current position but its for the first position with x and y 0
                    e, o = nextstate2(0, 0, np_arr[item],robot)
                    robot.drive_straight(distance_mm(210), speed_mmps(50)).wait_for_completed()
                elif np_arr[item] == current:
                    # Finding the Conditions with respect to the every point
                    find4conditions(e,o)
                    # Finding the next state of the current position but its for the first position with e and o of the previous values
                    e, o = nextstate2(e, o, np_arr[item],robot)
                    robot.drive_straight(distance_mm(210), speed_mmps(50)).wait_for_completed()
                else:
                    # Finding the Conditions with respect to the every point
                    find4conditions(e, o)
                    # Finding the next state of the current position but its for the first position with e and o of the previous values
                    e, o = nextstate2(e, o, np_arr[item],robot)
                    destination = np_arr[item]
                    #For the rotation of the robot according to the direction
                    rotation = destination - current
                    print(rotation)
                    robot.turn_in_place(degrees(rotation)).wait_for_completed()
                    robot.drive_straight(distance_mm(190), speed_mmps(50)).wait_for_completed()
                    current = destination
        rows=0
        #for finding the number of columns
        with open("file.txt") as f:
            line = f.readline()
            columns=len(line.split())
        # print(columns,"columns")
        #for finding the number of the rows
        with open("file.txt") as foo:
            for line in foo:
                rows=rows+1
            # print(rows,"rows")
        #for insertig the file values into the 2D array
        with open("file.txt") as textFile:
            lines = [line.split() for line in textFile]
        print(lines)




        #for finding the next state with respect to the current position
        def nextstate2(x,y,Action,robot: cozmo.robot.Robot):
            #It depends on the Action to be performed if it is right we add the y value
            if Action == 360:
                y = y + 1
                if y >= columns or y < 0:
                    #if the y is greater then the columns or less than 0 for the top left and the top right corner
                    robot.say_text("There is No Path Here").wait_for_completed()
                    # cozmo.run_program(cozmo_program1)
                    print("There is No Path Here")
                else:

                    #if the next direction the child taken is where there is block then the system stops and the childre
                    #must start its game again
                    if lines[x][y]=='B':
                        robot.say_text("Here is the path blocked").wait_for_completed()
                        # cozmo.run_program(cozmo_program1)
                        print("Here is the path blocked")
                        sys.exit()
                    else:
                        print("I am here ",lines[x][y])

            elif Action == 270:
                x = x + 1
                if x >= rows or x < 0:
                    robot.say_text("There is no path here").wait_for_completed()
                    # cozmo.run_program(cozmo_program1)
                    print("There is No Path Here")
                    sys.exit()
                else:
                    if lines[x][y]=='B':
                        robot.say_text("Here is the path blocked").wait_for_completed()
                        # cozmo.run_program(cozmo_program1)
                        print("Here is the path blocked")
                        sys.exit()
                    else:
                        print("I am here ",lines[x][y])

            elif Action == 90:
                x = x - 1
                if x >= rows or x < 0:
                    robot.say_text("There is no path here").wait_for_completed()
                    # cozmo.run_program(cozmo_program1)
                    print("There is No Path Here")
                    sys.exit()
                else:
                    if lines[x][y]=='B':
                        robot.say_text("Here is the path blocked").wait_for_completed()
                        # cozmo.run_program(cozmo_program1)
                        print("Here is the path blocked")
                        sys.exit()
                    else:
                        print("I am here ",lines[x][y])

            elif Action == 180:
                y = y - 1
                if y >= columns or y < 0:
                    robot.say_text("There is no path here").wait_for_completed()
                    # cozmo.run_program(cozmo_program1)
                    print("There is No Path Here")
                    sys.exit()
                else:
                    if lines[x][y]=='B':
                        robot.say_text("Here is the path blocked").wait_for_completed()
                        # cozmo.run_program(cozmo_program1)
                        print("Here is the path blocked")
                        sys.exit()
                    else:
                        print("I am here ",lines[x][y])

            return x, y


        def find4conditions(x,y):
            #It is without adding the 1 in y
            #If the robot move out of the bondary as expected in the case of top right y>columns
            if y>columns or y<0:
                robot.say_text("There is No Path Here").wait_for_completed()
                # cozmo.run_program(cozmo_program1)
                print("There is No Path Here")
                sys.exit()
            # It is without adding the 1 in x
            # If the robot move out of the bondary as expected in the case of bottom left x>rows
            if x>rows or x<0:
                robot.say_text("There is No Path Here").wait_for_completed()
                # cozmo.run_program(cozmo_program1)
                print("There is No Path Here")
                sys.exit()
            #for right word
            #For printing current position
            print(lines[x][y],"The current position of the robot")
            y=y+1
            if y>=columns:
                #FOr the next state if it is out of the boundary
                print("Null"," is the Right Word with respect to the current position")
            else:
                right_word=lines[x][y]
                # if right_word=='B':
                    # sys.exit()

                print(right_word,"is the Right Word with respect to the current position")
            y = y - 1
            #for left word

            y=y-1
            if y<0:
                #After y=y-1 if it is negative for the Top left and the bottom left
                print("Null","Left Word")
            else:
                 left_word=lines[x][y]
                 print(left_word," is the Left Word with respect to the current position")
            y = y + 1
            #for up word
            x=x-1
            if x<0:
                #for the top left
                print("Null","Up Word")
            else:
                 up_word=lines[x][y]
                 print(up_word,"is the Up Word with repsect to the current position")
            x = x + 1
            #for down word
            x=x+1
            if x>=rows:
                print("Null","Down Word")
            else:
                down_word=lines[x][y]
                print(down_word,"is the Down Word with respect to the current position")
            x = x - 1
            return x,y

        # def cozmo_program1(robot:cozmo.robot.Robot):
        #     robot.say_text("Hello World").wait_for_completed()

        cozmo.run_program(cozmo_program)


        #    if className == "down":
         #        print("Predicted: downward")
         #        cozmo.run_program(cozmo_down_program)
         #
         #    elif className == "Down_Left":
         #        print("Predicted: downleft")
         #        cozmo.run_program(cozmo_left_program)
         #    elif answer == "down_right":
         #        print("Predicted: Down Right")
         #        cozmo.run_program(cozmo_right_program)
         #
         #    elif answer == "Left":
         #        print("Predicted: Left")
         #        cozmo.run_program(cozmo_up_program)
         #
         #    elif answer == "Left_Down":
         #        print("Predicted: Left Down")
         #        cozmo.run_program(cozmo_up_program)
         #
         #    elif answer == "Right":
         #        print("Predicted: Right")
         #        cozmo.run_program(cozmo_up_program)
         #
         #    elif answer == "Right_Down":
         #        print("Predicted: Right Down")
         #        cozmo.run_program(cozmo_up_program)
         #    elif answer == "Right_Up":
         #        print("Predicted: Right Up")
         #        cozmo.run_program(cozmo_up_program)
         #
         #
         # # dataset = tf.data.Dataset.from_tensor_slices((files, labels))
         # # img_mask = 'FinalTesting/*.png'
         # # img_names = glob(img_mask)
         # #
         # #
         # # for fn in img_names:
         # #     img = cv2.imread(fn)
         # #     print('processing %s...' % fn, )
         # #
         # #
         # #
         # #
         # #     test_image = image.load_img(fn, target_size=(80, 80))
         # #     test_image = image.img_to_array(test_image)
         # #     test_image = np.expand_dims(test_image, axis=0)
         # #     array = classifier.predict(test_image)
         # #     # array = pred.sees(test_image)
         # #     result = array[0]
         # #     #    training_set.class_indices
         # #     # train_generator.class_indices
         # #     # print(result)
         # #     answer = np.argmax(result)
         # #
         # #     if answer == 0:
         # #         print("Predicted: downward")
         # #     #        cozmo.run_program(cozmo_down_program)
         # #
         # #     elif answer == 1:
         # #         print("Predicted: downleft")
         # #     #        cozmo.run_program(cozmo_left_program)
         # #     elif answer == 2:
         # #         print("Predicted: Down Right")
         # # #        cozmo.run_program(cozmo_right_program)
         # #

        # # i = 1
        # # mystring = []
        # mystring2 = []
        # list1 = []
        # # contents = f.read()
        # # mystring.append(contents)
        #
        # # for line in contents:
        # #     for c in line.split():
        # #         mystring2.append(c)
        # # print(mystring2)
        # # for index in range(len(mystring2)):
        #     # The current index value
        #     indexw = index
        #     if mystring2[indexw] == '/':
        #         continue
        #     else:
        #         print(mystring2[indexw])
        #         list1.append(mystring2[indexw])
        #
        #     # for right value with respect to that index
        #     indexu = index + 1
        #     if mystring2[indexu] == '/':
        #         print("Null")
        #         list1.append("Null")
        #     else:
        #         print(mystring2[indexu])
        #         list1.append(mystring2[indexu])
        #         index = indexu - 1
        #     # for down value with respect to that index
        #
        #     indexc = index + 9
        #     if mystring2[indexc] == '/':
        #         list1.append("Null")
        #         print("Null")
        #     else:
        #         print(mystring2[indexc])
        #         list1.append(mystring2[indexc])
        #         index = indexc - 9
        #
        #     # for upward value with respect to that index
        #     indext = index - 9
        #     if mystring2[indext] == '/':
        #         list1.append("Null")
        #         print("Null")
        #     else:
        #         print(mystring2[indext])
        #         list1.append(mystring2[indext])
        #         index = indext + 9
        #     # for left value with respect to that index
        #     indexf = index - 1
        #     if mystring2[indexf] == '/':
        #         list1.append("Null")
        #         print("Null")
        #     else:
        #         print(mystring2[indexf])
        #         list1.append(mystring2[indexf])
        #         index = indexf + 1
        #     print(list1)
        #     list1.clear()
