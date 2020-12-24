# Tile renderer. Moving blocks
import math
import pygame as pg
from numpy import exp, transpose, dot
from time import sleep
from random import randint, getrandbits, choice, choices, sample


class Tile:
    def __init__(self, xpos, ypos, isCar, state, i, j, speed=0, direction='nan', brain=0, moveCounter=[0, 0, 0], previewsPosition=[None, None]):
        self.xpos = xpos
        self.ypos = ypos
        self.state = state # 0: Road, 1: Wall, 2: Crashed
        self.isCar = isCar # True: it's a car, False: it's not a car
        self.i = i # An integer showing the node in the x axis
        self.j = j # An integer showing the node in the y axis
        self.speed = speed # a float that describes the speed of movement
        self.direction = direction # 'nan' 'up', 'down', 'left', 'right' a string that describes the direction of movement
        self.brain = brain # the feed forward neural network
        self.moveCounter = moveCounter # counts the number of moves taken until a crash
        self.previewsPosition = previewsPosition # the i and j of the previews position

    def getI(self):
        return self.i
    def getJ(self):
        return self.j
    def getX(self):
        return self.xpos
    def getY(self):
        return self.ypos
    def getisCar(self):
        return self.isCar   
    def getState(self):
        return self.state
    def getSpeed(self):
        return self.speed
    def getDirection(self):
        return self.direction
    def getBrain(self):
        return self.brain
    def getMoveCounter(self):
        return self.moveCounter
    def getPreviewsPosition(self):
        return self.previewsPosition

    def setX(self, xpos):
        self.xpos = xpos
    def setY(self, ypos):
        self.ypos = ypos
    def setisCar(self, isCar):
        self.isCar = isCar  
    def setState(self, state):
        self.state = state
    def setSpeed(self, speed):
        self.speed = speed
    def setDirection(self, direction):
        self.direction = direction
    def setBrain(self, brain):
        self.brain = brain
    def setMoveCounter(self, moveCounter):
        self.moveCounter = moveCounter
    def setPreviewsPosition(self, previewsPosition):
        self.previewsPosition = previewsPosition

class Brain(object):
    def __init__(self, sizes, newGen=False, oldBiases=[], oldWeights=[], numOfBrainChanges=20):
        self.numLayers = len(sizes)
        self.sizes = sizes
        self.numOfBrainChanges = numOfBrainChanges
        numOfWeights = 0
        for i in range(self.numLayers-1):
            numOfWeights += self.sizes[i] * self.sizes[i+1]
        self.numOfWeights = numOfWeights
        if not newGen:
            self.biases = [randint(0,10)/10 for i in sizes[1:]]

            a = [i+1 for i in range(self.numLayers-1)]
            b = [i for i in range(self.numLayers-1)]
            row = []
            col = []
            oldWeights = []

            for i,j in zip(a,b):
                for y in range(self.sizes[j]):
                    for x in range(self.sizes[i]):
                        row.append(randint(0,10)/10)
                    col.append(row)
                    row = []
                oldWeights.append(col)
                col = []
            self.weights = oldWeights
        else:
            offsetBiasSample = [randint(-10,10)/100 for i in range(10)]
            offsetBias = choice(offsetBiasSample)
            index_offsetBias = randint(0,len(oldBiases)-1)
            oldBiases[index_offsetBias] = oldBiases[index_offsetBias] + offsetBias 
            oldBiases = self.fixNewBias(oldBiases)

            a = [i+1 for i in range(self.numLayers-1)]
            b = [i for i in range(self.numLayers-1)]
            row = []
            col = []
            newWeights = []
            stash = [randint(-10,10)/100 for i in range(self.numOfBrainChanges)] + [0]*(self.numOfWeights-self.numOfBrainChanges)
            c = 0
            for i,j in zip(a,b):
                for y in range(self.sizes[j]):
                    for x in range(self.sizes[i]):
                        row.append(oldWeights[c][y][x] + choice(stash))
                    col.append(row)
                    row = []
                newWeights.append(col)
                col = []
                c += 1

            newerWeights = self.fixNewWeights(newWeights)

            self.biases = oldBiases
            self.weights = newerWeights

    def sigmoid(self, z):
        return [1 / (1 + exp(-i)) for i in z]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            w = transpose(w)
            a = self.sigmoid(dot(w, a) + b)
        return self.giveDirection(a)

    def giveDirection(self, a):
        a = list(a)
        indexA = a.index(max(a))
        if indexA == 0: return 'up'
        elif indexA == 1: return 'down'
        elif indexA == 2: return 'left'
        elif indexA == 3: return 'right'
        else:
            return 'nan'#choice(['up', 'down', 'left', 'right'])
    
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases

    def fixNewBias(self, x):
        newX = []
        for i in x:
            i = round(i, 3)
            if i > 1:
                newX.append(1+(1-i))
            elif i < 0:
                newX.append(1-(1+i))
            else:
                newX.append(i)
        return newX

    def fixNewWeights(self, newWeights):
        row = []
        col = []
        weights = []
        for i in newWeights:
            for j in i:
                for z in j:
                    if z > 1:
                        row.append(round(1+(1-z),2))
                    elif z < 0:
                        row.append(round(1-(1+z),2))
                    else:
                        row.append(round(z,2))
                col.append(row)
                row = []
            weights.append(col)
            col = []
        return weights
            

class Grid:
    SCREEN_WIDTH = 500 # width (in px)
    SCREEN_HEIGHT = 500 # height (in px)
    TileWidth = 1 # initializing the width of tile (in px)
    TileHeight = 1 # initializing the height of tile (in px)
    tilesMatrix = [] # initializing the matrix of Tile instances
    listOfMovesCounters = [] # initializing the vector that holds the last moveCounter value at the time of crash for each generation (each generation it resets)  
    brainSize = [4,4,4]
    lastBestBrain = None
    WHITE=(255,255,255)
    BLUE=(0,0,255)
    BLACK=(0,0,0)
    RED = (255,0,0)
    tileMapState = [[1]*10,
        [1,0,0,0,0,0,0,0,1,1],  
        [1,0,0,0,1,1,0,0,0,1],
        [1,0,0,0,1,1,1,0,0,1],
        [1,0,0,1,1,1,1,0,0,1], 
        [1,0,0,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1], 
        [1,1,0,0,0,0,0,0,0,1], 
        [1]*10]

    def __init__(self, ni, nj):
        self.grid = [[0]*ni,[0]*nj] # Number of nodes in x and y direction
        self.ni = ni
        self.nj = nj
        pg.init()
        self.WIN = pg.display.set_mode((Grid.SCREEN_WIDTH, Grid.SCREEN_HEIGHT), pg.RESIZABLE) # creates a screen of 600px X 800px
        pg.display.set_caption('Tile Renderer V02 - Moving Blocks V02')
        self.font = pg.font.Font(None, 25)
        for i in range(self.ni):
            tempLst = []
            for j in range(self.nj):
                tempLst.append(Tile(xpos=i * Grid.TileWidth, ypos=j * Grid.TileHeight, isCar=False, state=Grid.tileMapState[i][j], i=i, j=j)) #  getrandbits(1)
            Grid.tilesMatrix.append(tempLst)
        
        self.spawnCars()
        self.GameLoop()

    def spawnCars(self, bestBrain=None, ):
        # Spawn the cars 
        numberOfCars = 2
        groupX = [1,2,3]#[i+1 for i in range(self.ni-2)]
        groupY = [1,2,3]#[i+1 for i in range(self.nj-2)]
        if bestBrain == None:
            for i in range(numberOfCars):#len(groupY)
                x = choice(groupX)
                groupX.pop(groupX.index(x))
                y = choice(groupY)
                groupY.pop(groupY.index(y))
                Grid.tilesMatrix[x][y].setisCar(True)
                Grid.tilesMatrix[x][y].setSpeed(1)
                Grid.tilesMatrix[x][y].setDirection(self.chooseNewDirection())
                Grid.tilesMatrix[x][y].setPreviewsPosition([x, y])
                Grid.tilesMatrix[x][y].setBrain(Brain(sizes=Grid.brainSize, newGen=False, oldBiases=[0], oldWeights=[0]))
                Grid.lastBestBrain = Grid.tilesMatrix[x][y].getBrain()
                Grid.listOfMovesCounters.append([0, x, y])
        else:
            bestWeights = bestBrain.getWeights()
            bestBiases = bestBrain.getBiases()
            for i in range(numberOfCars):
                x = choice(groupX)
                groupX.pop(groupX.index(x))
                y = choice(groupY)
                groupY.pop(groupY.index(y))
                Grid.tilesMatrix[x][y].setisCar(True)
                Grid.tilesMatrix[x][y].setSpeed(1)
                Grid.tilesMatrix[x][y].setDirection(self.chooseNewDirection())
                Grid.tilesMatrix[x][y].setPreviewsPosition([x, y])
                Grid.tilesMatrix[x][y].setBrain(Brain(sizes=Grid.brainSize, newGen=True, oldBiases=bestBiases, oldWeights=bestWeights))
                Grid.lastBestBrain = Grid.tilesMatrix[x][y].getBrain()
                Grid.listOfMovesCounters.append([0, x, y])

    def initialize(self):    
        self.font = pg.font.Font(None, round(Grid.SCREEN_HEIGHT/10))
        Grid.TileHeight = Grid.SCREEN_HEIGHT / self.nj
        Grid.TileWidth = Grid.SCREEN_WIDTH / self.ni
        for i in range(self.ni):
            tempLst = []
            for j in range(self.nj):
                Grid.tilesMatrix[i][j].setX(i * Grid.TileWidth)
                Grid.tilesMatrix[i][j].setY(j * Grid.TileHeight)
        
    def GameLoop(self):
        running = True
        while (running):
            self.moveCar()
            self.drawGrid()
            ev = pg.event.get() # get all events
            for event in ev:
                if event.type == pg.MOUSEBUTTONUP:
                    pos = pg.mouse.get_pos() # x and y
                    running = True 
                if event.type == pg.QUIT:
                    running = False
                if event.type == pg.VIDEORESIZE:
                    self.WIN = pg.display.set_mode((event.w, event.h), pg.RESIZABLE)
                    Grid.SCREEN_WIDTH = event.w
                    Grid.SCREEN_HEIGHT = event.h
                    self.initialize()
            if running is None: 
                running = True
            pg.display.update() # updates the screen
            #print(Grid.listOfMovesCounters)
            sleep(0.01)

    def checkArround(self, i, j):
        back = []
        if 0 < i < self.ni-1 and 0 < j < self.nj-1:
            if (Grid.tilesMatrix[i+1][j].getState() == 1 or Grid.tilesMatrix[i+1][j].getisCar()):
                back.append(1)
            else:
                back.append(0)
            if (Grid.tilesMatrix[i-1][j].getState() == 1 or Grid.tilesMatrix[i-1][j].getisCar()):
                back.append(1)
            else:
                back.append(0)
            if (Grid.tilesMatrix[i][j+1].getState() == 1 or Grid.tilesMatrix[i][j+1].getisCar()):
                back.append(1)
            else:
                back.append(0)
            if (Grid.tilesMatrix[i][j-1].getState() == 1 or Grid.tilesMatrix[i][j-1].getisCar()):
                back.append(1)
            else:
                back.append(0)
            #print(f'The checkArround returns: {back}')
        return back
  
    def checkForCrash(self, car):
        if (car.getisCar() and (car.getState() == 1 or car.getState() == 2)):
            # if the counter of moves is the biggest then copy its brain to the next generation
            return True
        else:
            return False

    def chooseNewDirection(self):
        direction = ['up', 'down', 'left', 'right']
        return choice(direction)

    def moveCar(self):
        tempListofCars = []
        for i in range(self.ni):
            for j in range(self.nj):
                tempCar = Grid.tilesMatrix[i][j]
                if tempCar.getisCar(): 
                    tempListofCars.append(tempCar)
        if (len(tempListofCars) == 0): 
            # change this to get the max from all the cars or greater than 2
            try:
                if Grid.listOfMovesCounters[-1][0] > Grid.listOfMovesCounters[-2][0]:
                    i = Grid.listOfMovesCounters[-1][1]
                    j = Grid.listOfMovesCounters[-1][2]
                    best = Grid.listOfMovesCounters[-1][0]
                else:
                    i = Grid.listOfMovesCounters[-2][1]
                    j = Grid.listOfMovesCounters[-2][2]
                    best = Grid.listOfMovesCounters[-2][0]
                print(f'the best brains i, j = ({i}, {j}) with {best} steps!')
                bestBrain = Grid.tilesMatrix[i][j].getBrain()
            except:
                # replace with the last recorded best brain
                bestBrain = Grid.lastBestBrain
            self.spawnCars(bestBrain)
            Grid.listOfMovesCounters = []
        for car in tempListofCars:
            i = car.getI()
            j = car.getJ()
            if self.checkForCrash(car):
                Grid.tilesMatrix[i][j].setisCar(False)
                Grid.tilesMatrix[i][j].setState(2)
                Grid.tilesMatrix[i][j].setSpeed(0) 
                Grid.tilesMatrix[i][j].setDirection('nan')
                Grid.tilesMatrix[i][j].setPreviewsPosition([i, j])
                # change i and j to the previus position
                Grid.listOfMovesCounters.append([car.getMoveCounter()[0], i, j])
            else: 
                if (i!=0 and i!=self.ni-1 and j!=0 and j!=self.nj-1):
                    # Do something if they reach the boundaries
                    arroundState = self.checkArround(i, j)
                    if car.getDirection() == 'left':
                        Grid.tilesMatrix[i-1][j].setisCar(True)
                        Grid.tilesMatrix[i][j].setisCar(False)
                        Grid.tilesMatrix[i-1][j].setSpeed(car.getSpeed())
                        Grid.tilesMatrix[i][j].setSpeed(0) 
                        Grid.tilesMatrix[i-1][j].setDirection(car.getBrain().feedforward(arroundState))
                        Grid.tilesMatrix[i][j].setDirection('nan')  
                        Grid.tilesMatrix[i-1][j].setBrain(car.getBrain())
                        Grid.tilesMatrix[i][j].setBrain(None) 
                        Grid.tilesMatrix[i-1][j].setMoveCounter([car.getMoveCounter()[0]+1, i-1, j])
                        Grid.tilesMatrix[i][j].setMoveCounter([0, 0, 0])  
                        Grid.tilesMatrix[i-1][j].setPreviewsPosition(car.getPreviewsPosition())
                        Grid.tilesMatrix[i][j].setPreviewsPosition([None, None])                                                                    
                    if car.getDirection() == 'right':
                        Grid.tilesMatrix[i+1][j].setisCar(True)
                        Grid.tilesMatrix[i][j].setisCar(False)
                        Grid.tilesMatrix[i+1][j].setSpeed(car.getSpeed())
                        Grid.tilesMatrix[i][j].setSpeed(0) 
                        Grid.tilesMatrix[i+1][j].setDirection(car.getBrain().feedforward(arroundState)) # car.getDirection()self.chooseNewDirection()
                        Grid.tilesMatrix[i][j].setDirection('nan') 
                        Grid.tilesMatrix[i+1][j].setBrain(car.getBrain())
                        Grid.tilesMatrix[i][j].setBrain(None)  
                        Grid.tilesMatrix[i+1][j].setMoveCounter([car.getMoveCounter()[0]+1, i+1, j])
                        Grid.tilesMatrix[i][j].setMoveCounter([0, 0, 0])   
                        Grid.tilesMatrix[i+1][j].setPreviewsPosition(car.getPreviewsPosition())
                        Grid.tilesMatrix[i][j].setPreviewsPosition([None, None])                                                                           
                    if car.getDirection() == 'up':
                        Grid.tilesMatrix[i][j-1].setisCar(True)
                        Grid.tilesMatrix[i][j].setisCar(False) 
                        Grid.tilesMatrix[i][j-1].setSpeed(car.getSpeed())
                        Grid.tilesMatrix[i][j].setSpeed(0) 
                        Grid.tilesMatrix[i][j-1].setDirection(car.getBrain().feedforward(arroundState))
                        Grid.tilesMatrix[i][j].setDirection('nan')  
                        Grid.tilesMatrix[i][j-1].setBrain(car.getBrain())
                        Grid.tilesMatrix[i][j].setBrain(None)
                        Grid.tilesMatrix[i][j-1].setMoveCounter([car.getMoveCounter()[0]+1, i, j-1])
                        Grid.tilesMatrix[i][j].setMoveCounter([0, 0, 0]) 
                        Grid.tilesMatrix[i][j-1].setPreviewsPosition(car.getPreviewsPosition())
                        Grid.tilesMatrix[i][j].setPreviewsPosition([None, None])                                                                                                
                    if car.getDirection() == 'down':
                        Grid.tilesMatrix[i][j+1].setisCar(True)
                        Grid.tilesMatrix[i][j].setisCar(False) 
                        Grid.tilesMatrix[i][j+1].setSpeed(car.getSpeed())
                        Grid.tilesMatrix[i][j].setSpeed(0)  
                        Grid.tilesMatrix[i][j+1].setDirection(car.getBrain().feedforward(arroundState))
                        Grid.tilesMatrix[i][j].setDirection('nan') 
                        Grid.tilesMatrix[i][j+1].setBrain(car.getBrain())
                        Grid.tilesMatrix[i][j].setBrain(None)    
                        Grid.tilesMatrix[i][j+1].setMoveCounter([car.getMoveCounter()[0]+1, i, j+1])
                        Grid.tilesMatrix[i][j].setMoveCounter([0, 0, 0])  
                        Grid.tilesMatrix[i][j+1].setPreviewsPosition(car.getPreviewsPosition())
                        Grid.tilesMatrix[i][j].setPreviewsPosition([None, None])                                                   

    def drawGrid(self):
        for i in range(self.ni):
            for j in range(self.nj):
                pg.draw.rect(self.WIN,Grid.WHITE,(int(Grid.tilesMatrix[i][j].getX()),int(Grid.tilesMatrix[i][j].getY()),int(Grid.TileWidth),int(Grid.TileHeight)))
                if (Grid.tilesMatrix[i][j].getState() == 0):
                    color = Grid.BLACK
                elif (Grid.tilesMatrix[i][j].getState() == 1):
                    color = Grid.BLUE
                if (Grid.tilesMatrix[i][j].getState() == 2):
                    color = Grid.RED
                if Grid.tilesMatrix[i][j].getisCar():
                    color = Grid.WHITE

                pg.draw.rect(self.WIN,color,(int(Grid.tilesMatrix[i][j].getX()+1),int(Grid.tilesMatrix[i][j].getY()+1),int(Grid.TileWidth-1),int(Grid.TileHeight-1)))
                


newGrid = Grid(9,10) # ni, nj
