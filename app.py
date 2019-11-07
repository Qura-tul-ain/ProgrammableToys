from flask import Flask, render_template, request, jsonify, redirect, url_for
import random, json
import numpy as np
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps
app = Flask(__name__)


# First we will get data from js file then store data in word
# replace spaces with 1
# apply condition base on 1 ,seprate the words(up,right,down...)
# and then store in final array after reversing .
##@app.route('/get_data', methods=['GET','POST'])
##def data():
##    data=request.form['javascript_data']
####    data = request.get_json()
##    result_array = np.empty((0, 100))
##    result='' # store data from js file
##    dataarray=[]# store new array having 1 instead of space
##    for item in data:
##        result +=item
##
##    count=0
##    word=''
##    # rerplace space with 1
##    for a in result:
##        if (a.isspace()) == False:
##            dataarray.extend(a)
##            count+=1
##        else:
##            dataarray.extend('1')
##            count+=1
##
##
####
####    for x in range(len(dataarray)):
####        print(dataarray[x])
##    index=0
##    # store words 
##    finalArray = []
##    for y in range(len(dataarray)):
##        if dataarray[y]=='1':
##
##            for x in range(index, y):
##                word= dataarray[x] + word
##                index=y+1
###           reverse strig    
##            word=word[::-1]
##            # store in final array
##            finalArray.append(word)
##            word=''
##      # run loop om final array and apply conditions.      
##    for c in range(len(finalArray)):
##        if finalArray[c]=='Up':
##            print("up")
##            cozmo.run_program(program)
##        elif finalArray[c]=='down':
##            print("down")
##            cozmo.run_program(program)
##        elif finalArray[c]=='Right':
##            print("right")
##            cozmo.run_program(Right)
##        else:
##            print("left")
##            cozmo.run_program(Left)
##
##
##
##    return render_template('blockly.html')
##
##
##
##@app.route('/program', methods=['GET','POST'])
##def program(robot: cozmo.robot.Robot):
##        
##      # Drive forwards for 150 millimeters at 50 millimeters-per-second.
##        robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
##       # Turn 90 degrees to the left.
##       # Note: To turn to the right, just use a negative number.
##       # robot.turn_in_place(degrees(90)).wait_for_completed()
##       
##
##
###move down_left				
###@app.route('/left',methods=['GET','POST'])		
##def Left(robot):
##        
##       #Drive forwards for 150 millimeters at 50 millimeters-per-second.
##        #robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
##        robot.turn_in_place(degrees(-90)).wait_for_completed()
##        robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()	
##
##
##
###move down_right
###@app.route('/right',methods=['GET','POST'])
##def Right(robot: cozmo.robot.Robot):
##     # Drive forwards for 150 millimeters at 50 millimeters-per-second.
## 
##        #robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()
##        robot.turn_in_place(degrees(90)).wait_for_completed()
##        robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()



@app.route('/StageOne',methods=['GET','POST'])
def StageOne():
        number=1
        return render_template('blockly.html',number=number)

@app.route('/StageTwo',methods=['GET','POST'])
def StageTwo():
        number=2
        return render_template('blockly.html',number=number)

@app.route('/StageThree',methods=['GET','POST'])
def StageThree():
        number=3
        return render_template('blockly.html',number=number)

@app.route('/StageFour',methods=['GET','POST'])
def StageFour():
        number=4
        return render_template('blockly.html',number=number)

@app.route('/StageFive',methods=['GET','POST'])
def StageFive():
        number=5
        return render_template('blockly.html',number=number)

@app.route('/StageSix',methods=['GET','POST'])
def StageSix():
        number=6
        return render_template('blockly.html',number=number)

@app.route('/StageSeven',methods=['GET','POST'])
def StageSeven():
        number=7
        return render_template('blockly.html',number=number)

@app.route('/StageEight',methods=['GET','POST'])
def StageEight():
        number=8
        return render_template('blockly.html',number=number)

@app.route('/StageNine',methods=['GET','POST'])
def StageNine():
        number=9
        return render_template('blockly.html',number=number)


@app.route('/StageTen',methods=['GET','POST'])
def StageTen():
        number=10
        return render_template('blockly.html',number=number)


@app.route('/')
def index():
        return render_template('homepage.html')




if __name__ == '__main__':
	app.run()







