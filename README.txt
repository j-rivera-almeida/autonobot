The DriveAPI is included in the DriveAPI.py. You will create the script to run the rover in the Run.py file. 
In Run.py there is an inherited class over Rover called MyRover. Your code will go in the Script function of MyRover. 
When calling functions in the Script function, you will need to use the rover object passed in. For example, rover.PutInDrive().

The DriveAPI offers the following functions:

Driving
****************
PressGas() - Presses the gas pedal on the rover. Make sure to call PressGas() after putting the rover in drive or reverse.
ReleaseGas() - Takes the foot off the gas and causes the rover to slow down. 
PutInDrive() - Makes the rover ready to drive forward. The rover must be put in drive or reverse before it will move. 
PutInReverse() - Makes the rover ready to drive backwards. The rover does not have a break. Instead, you can ReleaseGas() or PutInReverse() and drive to slow the rover. 
DriveFor(timeLength) - Causes the rover to drive for the stated timeLength, in seconds, before turning or changing gear.

Turning
****************
Note: Turn commands only alter which way the wheel is turned. The rover also needs to be put in drive or reverse, as well as pressing the gas, to move the rover. 
TurnLeft() - Turns the rover to the left. 
TurnRight() - Turns the rover to the right.
GoStraight() - Removes and turn direction and makes the rover head straight. 


Other
****************
Finish() - Call Finish once the rover is done driving. After Finish is called the rover will stop and the python script will end. 
Run() - Call Run in the main function of your code. This has been implemented for you. It will get the rover's internal state preparred and begin running your script.