import numpy as np
import math
import copy
import random
import kociemba #From https://pypi.org/project/kociemba/
#pip install kociemba to use

from numpy.lib import average
from check import rotateFace
from cmu_112_graphics import *

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

#I will add another page to the application when that will include all these instructions in a more concise and easily readable way after tp2

def appStarted(app):
    app.mode = "HomeScreen"
    app.rubiksCube = []
    app.miniCube = []
    app.colorRotate = ""
    app.angleRotate = math.pi/40
    app.timerDelay = 2
    app.rotFrame = 0
    app.matrix = [[["white"] * 3 for row in range(3)] for row in range(6)]
    app.pressCoordinates = []
    app.colors = ["white", "red", "green", "yellow", "orange", "blue"]
    app.colorButtons = ["red", "orange", "yellow", "green", "blue", "white"]
    app.buttonCoordinates = []
    app.selectedColor = "white"
    app.drawSolution = False
    for i in range(len(app.matrix)):
        app.matrix[i][1][1] = app.colors[i]
    SolveScreen_createCoordBoard(app)
    SolveScreen_createButtonCoords(app)
    GameScreen_makeRubiksCube(app)
    HomeScreen_makeRubiksCube(app)
    for cuboid in app.miniCube:
        cuboid.points = cuboid.rotateX(math.pi/8, cuboid.points)
        cuboid.points = cuboid.rotateY(math.pi/8, cuboid.points)
    for cuboid in app.rubiksCube:
        cuboid.points = cuboid.rotateX(math.pi/4, cuboid.points)
        cuboid.points = cuboid.rotateY(math.pi/4, cuboid.points)


class MatrixFunctions():
    def subtract(self, matrixA, matrixB):
        return np.subtract(matrixA, matrixB)
    def crossProduct(self, matrixA, matrixB):
        return np.cross(matrixA, matrixB)
    def getCol(self, i, matrix):
        column = []
        for j in range(len(matrix)):
                column.append(matrix[j][i])
        return column
    def transpose(self, matrix):
        return np.transpose(matrix)
    def translateVector(self, x, y, dx, dy):
        return x + dx, y + dy
    def matrixMultiply(self, matrixA, matrixB):
        return np.dot(matrixA, matrixB)
    def rotateX(self, x, shape):
        return self.matrixMultiply(
            [[1, 0, 0],
            [0, math.cos(x), - math.sin(x)], 
            [0, math.sin(x), math.cos(x)]], shape)
    def rotateY(self, y, shape):
        return self.matrixMultiply(
            [[math.cos(y), 0, math.sin(y)], 
            [0, 1,0], 
            [-math.sin(y), 0, math.cos(y)]], shape)
    def rotateZ(self, z, shape):
        return self.matrixMultiply(
            [[math.cos(z), math.sin(z), 0],
            [-math.sin(z), math.cos(z), 0], 
            [0, 0, 1]], shape)

class Cuboid(MatrixFunctions):
    def __init__(self, points, 
    type, colors
    ):
        self.type = type
        self.colors = colors
        self.points = points
        self.pointsToDraw = [[0, 1, 2, 4], [3, 1, 2, 7],
                             [5, 1, 4, 7], [6, 2, 4, 7]]
        self.faces = [(0,1,3,2), (7,6,4,5),(1,5,7,3),(0,4,6,2),(2,3,7,6),(0,1,5,4)]

    def average(M):
        avg = 0
        for i in M:
            avg += i
        return avg/len(M)
    
    def getPosition(self):
        avgX = sum(self.points[0])/8
        avgY = sum(self.points[1])/8
        avgZ = sum(self.points[2])/8
        return (avgX, avgY, avgZ)
    
    def getClosestPoint(self):
        maxX, maxY, maxZ = max(self.points[0]), max(self.points[1]), max(self.points[2])
        return (maxX, maxY, maxZ)
    
    def max(M):
        # best = -10000000
        # for i in M:
        #     if i > best:
        #         best = i
        return sorted(M)[-1]

    def whatToDraw(self):
        points = self.points
        toDraw = []
        for i in range(len(self.faces)):
            w,x,y,z = self.faces[i][0], self.faces[i][1], self.faces[i][2], self.faces[i][3]
            matrixA = self.subtract(self.getCol(w, points), self.getCol(x, points))
            matrixB = self.subtract(self.getCol(y, points), self.getCol(x, points))
            if i %2 == 0:
                crossProduct = np.multiply(self.crossProduct(matrixA, matrixB), -1)
            else:
                crossProduct = self.crossProduct(matrixA, matrixB)
            if crossProduct[2] >= 0:
                toDraw.append((w,x,y,z))
        return toDraw

    def drawCuboid(self, canvas, toDraw, width, height):
        for i in range(len(toDraw)):
            w,x,y,z = toDraw[i]
            canvas.create_polygon(
            self.translateVector(self.points[0][w], self.points[1][w], width, height),
            self.translateVector(self.points[0][x], self.points[1][x], width, height),
            self.translateVector(self.points[0][y], self.points[1][y], width, height),
            self.translateVector(self.points[0][z], self.points[1][z], width, height),
            fill = self.colors[i%len(self.colors)], outline = "black")
    
#Makes the mini cube on the front page
def HomeScreen_makeRubiksCube(app):
    points = [[-10, -10, -10, -10,  10,  10,  10,  10],
                [-10,  10, -10,  10, -10,  10, -10,  10],
                [-10, -10,  10,  10, -10,-10,  10,  10]]
    colors = [["blue", "yellow", "orange"], ["orange","yellow"], ["green", "yellow", "orange"],
              ["blue", "black","orange"],           ["orange"],   ["green", "black","orange"],
              ["blue", "white","orange"], ["orange", "white"], ["green", "white","orange"],
             
              ["blue","yellow"],            ["yellow"],             ["green","yellow"],
              ["blue"],                                                 ["green"],
              ["blue", "white"],            ["white"],              ["green","white"],

              ["blue", "yellow", "red"], ["red","yellow"],          ["green", "yellow", "red"],
              ["blue", "black","red"],           ["red"],           ["green", "black","red"],
              ["blue", "white","red"],   ["red", "white"],          ["green", "white", "red"]]
    translations = [(-20,-20,20), (0,-20,20), (20,-20,20),
                    (-20,0,20), (0,0,20), (20,0,20),
                    (-20,20,20), (0,20,20), (20,20,20),

                    (-20,-20,0), (0,-20,0), (20,-20,0),
                    (-20,0,0),              (20,0,0),
                    (-20,20,0), (0,20,0), (20,20,0),

                    (-20,-20,-20), (0,-20,-20), (20,-20,-20),
                    (-20,0,-20), (0,0,-20), (20,0,-20),
                    (-20,20,-20), (0,20,-20), (20,20,-20),
                    ]
    for i in range(len(translations)):
        x,y,z = translations[i]
        cuboidpoints= copy.deepcopy(points)
        color = colors[i]
        if (x == 0 and y == 0) or (x == 0 and z == 0) or (y == 0 and z == 0):
            cubeType = "center"
        elif (y != 0) and ((x == z) or (x == -z)):
            cubeType = "corner"
        else:
            cubeType = "edge"
        GameScreen_translate(app, cuboidpoints, x, y, z)
        app.miniCube.append(Cuboid(cuboidpoints, cubeType, color))

def HomeScreen_cuboidsTodraw(app):
    cubeCopy = copy.deepcopy(app.miniCube)
    cubeCopy.sort(key = lambda c : -c.getClosestPoint()[2])
    return cubeCopy

def HomeScreen_drawRubiksCube(app, canvas, sortedCube):
    for cuboid in sortedCube:
        cuboid.drawCuboid(canvas, cuboid.whatToDraw(), app.width/2, app.height/2-90)

def HomeScreen_mousePressed(app,event):
    width, height = app.width/2, app.height/2
    if width-300 < event.x < width -50 and height < event.y < height+100:
        app.mode = "GameScreen"
    if width+50 < event.x < width+300 and height < event.y < height +100:
        app.mode = "SolveScreen"

def HomeScreen_timerFired(app):
    for cube in app.miniCube:
        cube.points = cube.rotateX(0.05, cube.points)
        cube.points = cube.rotateY(0.05, cube.points)

def HomeScreen_redrawAll(app,canvas):
    width, height = app.width/2, app.height/2
    canvas.create_text(400, 100, text = "The Rubik's Cube", font = "Comicsans 90 bold")
    round_rectangle(canvas, width-300, height, width -50, height+100,fill = "#83b0ff", outline = "black")
    canvas.create_text((width-300+width -50)/2, (height + height+100)/2, text = "Play", font = "Comicsans 40 bold")
    round_rectangle(canvas, width+50, height, width +300, height+100,fill = "#83b0ff", outline = "black")
    canvas.create_text((width+300+width +50)/2, (height + height+100)/2, text = "Solve", font = "Comicsans 40 bold")
    round_rectangle(canvas, width-125, height+150, width + 125, height + 250, fill = "#83b0ff", outline = "black")
    canvas.create_text(width, (height+150+height + 250)/2, text = "Directions", font = "Comicsans 40 bold")
    HomeScreen_drawRubiksCube(app, canvas, HomeScreen_cuboidsTodraw(app))

def round_rectangle(canvas, x1, y1, x2, y2, radius=25, **kwargs):
    points = [x1+radius, y1, x1+radius, y1, x2-radius, y1, x2-radius, y1, x2, y1,
              x2, y1+radius, x2, y1+radius, x2, y2-radius, x2, y2-radius, x2, y2,
              x2-radius, y2, x2-radius, y2,x1+radius, y2, x1+radius, y2, x1, y2,
              x1, y2-radius, x1, y2-radius, x1, y1+radius, x1, y1+radius,x1, y1]
    return canvas.create_polygon(points, **kwargs, smooth=True)

#Main Rubiks cube in the play page
def GameScreen_makeRubiksCube(app):
    points = [[-40, -40, -40, -40,  40,  40,  40,  40],
                [-40,  40, -40,  40, -40,  40, -40,  40],
                [-40, -40,  40,  40, -40,-40,  40,  40]]
    colors = [["blue", "yellow", "orange"], ["orange","yellow"], ["green", "yellow", "orange"],
              ["blue", "black","orange"],           ["orange"],   ["green", "black","orange"],
              ["blue", "white","orange"], ["orange", "white"], ["green", "white","orange"],
             
              ["blue","yellow"],            ["yellow"],             ["green","yellow"],
              ["blue"],                                                 ["green"],
              ["blue", "white"],            ["white"],              ["green","white"],

              ["blue", "yellow", "red"], ["red","yellow"],          ["green", "yellow", "red"],
              ["blue", "black","red"],           ["red"],           ["green", "black","red"],
              ["blue", "white","red"],   ["red", "white"],          ["green", "white", "red"]]

    translations = [(-80,-80,80), (0,-80,80), (80,-80,80),
                    (-80,0,80), (0,0,80), (80,0,80),
                    (-80,80,80), (0,80,80), (80,80,80),

                    (-80,-80,0), (0,-80,0), (80,-80,0),
                    (-80,0,0),              (80,0,0),
                    (-80,80,0), (0,80,0), (80,80,0),

                    (-80,-80,-80), (0,-80,-80), (80,-80,-80),
                    (-80,0,-80), (0,0,-80), (80,0,-80),
                    (-80,80,-80), (0,80,-80), (80,80,-80),
                    ]
    for i in range(len(translations)):
        x,y,z = translations[i]
        cuboidpoints= copy.deepcopy(points)
        color = colors[i]
        if (x == 0 and y == 0) or (x == 0 and z == 0) or (y == 0 and z == 0):
            cubeType = "center"
        elif (y != 0) and ((x == z) or (x == -z)):
            cubeType = "corner"
        else:
            cubeType = "edge"
        GameScreen_translate(app, cuboidpoints, x, y, z)
        app.rubiksCube.append(Cuboid(cuboidpoints, cubeType, color))

def GameScreen_cuboidsTodraw(app):
    cubeCopy = copy.deepcopy(app.rubiksCube)
    cubeCopy.sort(key = lambda c : -c.getClosestPoint()[2])
    return cubeCopy

def GameScreen_drawRubiksCube(app, canvas, sortedCube):
    for cuboid in sortedCube:
        cuboid.drawCuboid(canvas, cuboid.whatToDraw(), app.width/2, app.height/2)

def GameScreen_translate(app, M, x,y,z): #Adds to each x, y, z value of a matrix
    for i in range(len(M)):
        for j in range(len(M[0])):
            if i == 0:
                M[i][j] += x
            if i  == 1:
                M[i][j] += y
            if i == 2:
                M[i][j] += z

def isPointOnPlane(x0, y0, z0, x1, y1, z1): #checks if the points is on the same plane as the vector from x0,y0,z0
    if x0*(x0-x1) + y0*(y0-y1) + z0*(z0-z1) < 0.5:
        return True
    else:
        return False

def normalize(x,y,z): # Gets norm vector, from a point and the center
    norm = math.sqrt((x**2)+(y**2)+(z**2))
    return (x/norm, y/norm, z/norm)

def GameScreen_rotateFace(app, color, angle):
    rotationPoint = (0,0,0)
    for cuboid in app.rubiksCube:
        if cuboid.type == "center" and color in cuboid.colors:
            rotationPoint = cuboid.getPosition()
    xt, yt, zt = rotationPoint
    x,y,z = rotationPoint
    norm = normalize(x,y,z)
    x,y,z = norm
    w = np.matrix([[0, -z, y], #Rodriguez Rotation Matrix
         [z, 0, -x],
         [-y, x, 0]])
    for cuboid in app.rubiksCube:
        xc, yc, zc = cuboid.getPosition()
        if isPointOnPlane(x,y,z, xc, yc, zc):
            GameScreen_translate(app, cuboid.points, -xt, -yt, -zt) #translate to origin
            rot = np.identity(3) + math.sin(angle) * w + 2*(math.sin(angle/2)**2)*w*w #Rodgriguez rotation formula
            cuboid.points = (rot * np.matrix(cuboid.points)).tolist() # apply rotation
            GameScreen_translate(app, cuboid.points, xt, yt, zt) #translate back

def GameScreen_scramble(app):
    rotateColor = ["white", "red", "green", "yellow", "orange", "blue"]
    angle = [-math.pi/2, math.pi/2]
    app.rotFrame = 0
    for move in range(20):
        print(move)
        randomColor = rotateColor[random.randint(0,5)]
        app.colorRotate = randomColor
        randomAngle = angle[random.randint(0,1)]
        app.angleRotate = randomAngle
        app.rotFrame = 1
        GameScreen_rotateFace(app, app.colorRotate, app.angleRotate)

def GameScreen_keyPressed(app, event):
    for cuboid in app.rubiksCube:
        if event.key == "Right":
            cuboid.points = cuboid.rotateY(0.1, cuboid.points)
        if event.key == "Left":
            cuboid.points = cuboid.rotateY(-0.1, cuboid.points)
        if event.key == "Up":
            cuboid.points = cuboid.rotateX(0.1,cuboid.points)
        if event.key == "Down":
            cuboid.points = cuboid.rotateX(-0.1,cuboid.points)
        if event.key == "r":
            if(app.rotFrame == 0):
                app.colorRotate = "red"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "R":
            if(app.rotFrame == 0):
                app.colorRotate = "red"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "o":
            if(app.rotFrame == 0):
                app.colorRotate = "orange"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "O":
            if(app.rotFrame == 0):
                app.colorRotate = "orange"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "w":
            if(app.rotFrame == 0):
                app.colorRotate = "white"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "W":
            if(app.rotFrame == 0):
                app.colorRotate = "white"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "y":
            if(app.rotFrame == 0):
                app.colorRotate = "yellow"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "Y":
            if(app.rotFrame == 0):
                app.colorRotate = "yellow"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "b":
            if(app.rotFrame == 0):
                app.colorRotate = "blue"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "B":
            if(app.rotFrame == 0):
                app.colorRotate = "blue"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "g":
            if(app.rotFrame == 0):
                app.colorRotate = "green"
                app.rotFrame = 20
                app.angleRotate = math.pi/40
        if event.key == "G":
            if(app.rotFrame == 0):
                app.colorRotate = "green"
                app.rotFrame = 20
                app.angleRotate = -math.pi/40
        if event.key == "l":
            GameScreen_scramble(app)
        
def GameScreen_timerFired(app):
    if(app.rotFrame != 0):
        app.rotFrame -= 1
        GameScreen_rotateFace(app, app.colorRotate, app.angleRotate)

def GameScreen_mousePressed(app, event):
    if app.width-125 < event.x < app.width -10 and 10 < event.y < 75:
        app.mode = "HomeScreen"

def GameScreen_redrawAll(app, canvas):
    round_rectangle(canvas, app.width-125, 10, app.width -10 , 75, fill = "#83b0ff", outline = "black")
    canvas.create_text((app.width-125 + app.width -10)/2, (10 + 75) /2, text = "Back", font = "Comicsans 20 bold")
    round_rectangle(canvas, app.width-175, app.height-125, app.width -25, app.height -25, fill = "#83b0ff", outline = "black")
    canvas.create_text((app.width-175 + app.width -25)/2, (app.height-125 + app.height -25)/2, text = "Scramble", font = "Comicsans 30 bold")
    GameScreen_drawRubiksCube(app, canvas, GameScreen_cuboidsTodraw(app))

def SolveScreen_drawFlatCube(app, canvas):
    for face in range(len(app.matrix)):
        starts = [(app.width/2 - app.width/5, app.height/2 - 7*app.height/20), (app.width/2, app.height/2 - 3*app.height/20), 
        (app.width/2 - app.width/5, app.height/2 - 3*app.height/20), (app.width/2 - app.width/5, app.height/2 + 1*app.height/20), 
        (app.width/2 - 2*app.width/5, app.height/2 - 3*app.height/20), (app.width/2 + app.width/5, app.height/2 - 3*app.height/20)]
        x, y = starts[face]
        for row in range(len(app.matrix[0])):
            for col in range(len(app.matrix[0][0])):
                (x0, y0, x1, y1) = SolveScreen_getCellBounds(app, row, col, x, y)
                canvas.create_rectangle(x0, y0, x1, y1, fill = app.matrix[face][row][col])

def SolveScreen_getCellBounds(app, row, col, startX, startY):
    gridWidth  = app.width/5
    gridHeight = app.height/5
    x0 = gridWidth * col / 3 + startX
    x1 =  gridWidth * (col+1) / 3 + startX
    y0 =  gridHeight * row / 3 + startY
    y1 =  gridHeight * (row+1) / 3 + startY
    return (x0, y0, x1, y1)

def SolveScreen_createCoordBoard(app):
    for face in range(6):
        starts = [(app.width/2 - app.width/5, app.height/2 - 7*app.height/20), (app.width/2, app.height/2 - 3*app.height/20), 
        (app.width/2 - app.width/5, app.height/2 - 3*app.height/20), (app.width/2 - app.width/5, app.height/2 + 1*app.height/20), 
        (app.width/2 - 2*app.width/5, app.height/2 - 3*app.height/20), (app.width/2 + app.width/5, app.height/2 - 3*app.height/20)]
        x, y = starts[face]
        for row in range(3):
            for col in range(3):
                (x0, y0, x1, y1) = SolveScreen_getCellBounds(app, row, col, x, y)
                app.pressCoordinates.append((x0, y0, x1, y1))
        
def SolveScreen_drawColorButtons(app, canvas):
    colors = ["red", "orange", "yellow", "green", "blue", "white"]
    index = app.width/12
    for color in colors:
        # canvas.create_rectangle(index, app.height - 100, index + app.width/12, app.height-50, fill = color)
        round_rectangle(canvas, index, app.height - 100, index + app.width/12, app.height-50, fill = color, outline = "black")
        index += app.width / 10

def SolveScreen_createButtonCoords(app):
    colors = ["red", "orange", "yellow", "green", "blue", "white"]
    index = app.width/12
    for i in range(len(colors)):
        app.buttonCoordinates.append((index, app.height-100, index + app.width/12, app.height-50))
        index += app.width / 10

def SolveScreen_cubeMatrixtoString(cubeMatrix):
    cubeString = ""
    for face in range(len(cubeMatrix)):
        for row in range(len(cubeMatrix[0])):
            for col in range(len(cubeMatrix[0][0])):
                cubeString += cubeMatrix[face][row][col]
    return cubeString

def SolveScreen_createSolution(app):
    solutionMatrix = [[[""] * 3 for row in range(3)] for row in range(6)]
    for face in range(len(app.matrix)):
        for row in range(len(app.matrix[0])):
            for col in range(len(app.matrix[0][0])):
                    solutionMatrix[face][row][col] = SolveScreen_convertColor(app.matrix[face][row][col])
    solutionString = SolveScreen_cubeMatrixtoString(solutionMatrix)
    try:
        if kociemba.solve(solutionString) == "R L U2 R L' B2 U2 R2 F2 L2 D2 L2 F2":
            return "Your Cube is already solved."
        return kociemba.solve(solutionString)
    except:
        return "OOPS. Looks like the cube you inputed doesn't exist, press back and try again."

def SolveScreen_convertColor(color):
    if color == "red":
        return "R"
    elif color == "white":
        return "U"
    elif color == "yellow":
        return "D"
    elif color == "green":
        return "F"
    elif color == "blue":
        return "B"
    elif color == "orange":
        return "L"

def SolveScreen_drawSolution(app, canvas):
    canvas.create_rectangle(app.width/40, 3*app.height/8, 39*app.width/40, 5*app.height/8, fill = "#83b0ff", )
    canvas.create_text(app.width/2, app.height/2, text = SolveScreen_createSolution(app), font = "Comicsans 20 bold")

def SolveScreen_mousePressed(app, event):
    if app.width-125 < event.x < app.width -10 and 10 < event.y < 75:
        app.mode = "HomeScreen"
        appStarted(app)
    if app.width-250 < event.x < app.width-60 and app.height -200 < event.y < app.height- 100:
        app.drawSolution = True
    for i in range(len(app.buttonCoordinates)):
        (x0,y0,x1,y1) = app.buttonCoordinates[i]
        if x0 < event.x < x1 and y0 < event.y < y1:
            app.selectedColor = app.colorButtons[i]
    for i in range(len(app.pressCoordinates)):
        (x0,y0,x1,y1) = app.pressCoordinates[i]
        if x0 < event.x < x1 and y0 < event.y < y1:
            if (i - 9*(i//9))//3 == 1 and  (i - 9*(i//9))%3 == 1 and (i - 9*(i//9))%3 == (i - 9*(i//9))//3:
                break
            else:
                app.matrix[i//9][(i - 9*(i//9))//3][(i - 9*(i//9))%3] = app.selectedColor

def SolveScreen_redrawAll(app,canvas):
    round_rectangle(canvas, app.width-125, 10, app.width -10 , 75, fill = "#83b0ff", outline = "black")
    canvas.create_text((app.width-125 + app.width -10)/2, (10 + 75) /2, text = "Back", font = "Comicsans 20 bold")
    round_rectangle(canvas, app.width-250, app.height -200, app.width-60, app.height- 100, fill = "#83b0ff", outline = "black")
    canvas.create_text((app.width -250 + app.width - 60)/2, (app.height-200 + app.height-100)/2, text = "Get Solution", font = "Comicsans 30 bold")
    SolveScreen_drawFlatCube(app, canvas)
    SolveScreen_drawColorButtons(app, canvas)
    if app.drawSolution == True:
        SolveScreen_drawSolution(app, canvas)
    canvas.create_text((app.width/2, app.height/10), text = "Selected Color: " + app.selectedColor, font = "Comicsans 20 bold")

#Solved Cube 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
'''
UUU 
UUU # White (Up) 
UUU 

RRR 
RRR # Red (Right)
RRR

FFF 
FFF # Green (Front)
FFF

DDD 
DDD # Yellow (Down)
DDD

LLL 
LLL # Orange (Left)
LLL

BBB 
BBB # Blue (Back)
BBB
'''

'''
|************|
             |*U1**U2**U3*|
             |************|
             |*U4**U5**U6*|
             |************|
             |*U7**U8**U9*|
             |************|
 ************|************|************|************
 *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
 ************|************|************|************
 *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
 ************|************|************|************
 *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
 ************|************|************|************
             |************|
             |*D1**D2**D3*|
             |************|
             |*D4**D5**D6*|
             |************|
             |*D7**D8**D9*|
             |************|'''

#For the solver
def cubeStringtoMatrix(cubeString):
    matrix = [[[""] * 3 for row in range(3)] for row in range(6)]
    for face in range(len(matrix)):
        for row in range(len(matrix[0])):
            for col in range(len(matrix[0][0])):
                matrix[face][row][col] = cubeString[face*9 + row*3 + col]
    return matrix

def cubeMatrixtoString(cubeMatrix):
    cubeString = ""
    for face in range(len(cubeMatrix)):
        for row in range(len(cubeMatrix[0])):
            for col in range(len(cubeMatrix[0][0])):
                cubeString += cubeMatrix[face][row][col]
    return cubeString

def makeAMove(cubeString, move):
    moves = ["U", "U'", "R", "R'", "F", "F'", "D", "D", "L", "L'", "B", "B'"]
    faceIndexes = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    index = moves.find("U")
    return index

runApp(width=800, height=600)