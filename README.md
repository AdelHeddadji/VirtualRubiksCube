# VirtualRubiksCube

Rubiks simulator/solver in python using the kociemba algorithm

#Rubiks Cube Simulator Instructions:
    #Arrow Keys rotate the entire cube
    #Moves:
        #r - moves red face counterclockwise
        #R - moves red face clockwise
        #o - moves orange face counterclockwise
        #O - moves orange face clockwise
        #w - moves white face counterclockwise
        #W - moves white face clockwise
        #y - moves yellow face counterclockwise
        #Y - moves yellow face clockwise
        #b - moves blue face counterclockwise
        #B - moves blue face clockwise
        #g - moves green face counterclockwise
        #G - moves green face clockwise

#Rubiks Cube Solver Instructions:
    # Hold the cube with the green center facing forward, the white center facing up, 
    # And the yellow center face down orange to the left and red to the right
    # Pick a color, then select the piece it corresponds to on the 2d cube representation
    # Note that if you misinput even just one square, the application will crash
    # Then press get Solution and a solution will appear. 
    # It prints standard cube move notation
    # Here's an image on what the moves mean
    # https://miro.medium.com/max/1400/1*9bEE_2lmMUS2SnoskDKrCw.png
    # Note that, like when inputting the solutions you want to hold the cube with the green facing you white on top, red to the right, etc.

    #Moves that can be made
        #R Red Clockwise,    R' = Red CounterClockwise
        #L Orange Clockwise, L' = Orange CounterClockwise
        #B Blue Clockwise,   B' = Blue CounterClockwise
        #F Green Clockwise,  F' = Green CounterClockwise
        #U White Clockwise,  U' = White CounterClockwise
        #D Yellow Clockwise, D' = Yellow CounterClockwise
