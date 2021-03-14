# Simple genetic evolution machine learning moving blocks through a maze
Build in Python 3.
## The aim
Is to create a setup where two cars move in a maze with a feed forward neural network (NN) at the steering wheel. The NN is trained over many generations, every new generation, two cars are spawned with a brain structure inherited from the best performing car from the previews generation. When a car collides with a wall it stops. The goal is to see the cars develop a good strategy for staying in game as long as possible, meaning they do not collide to the walls, and go around the maze.
The tile rendering is given as a separate repository here:
```sh
https://github.com/rocku0/Tile-renderer
```
### Actual implementation
A tile world is created that creates a maze of walls and roads. Each car is controlled by a simple neural network with weights and biases randomly initialized with a range of [0, 1]. The NN uses the sigmoid activation funcion. The size of the NN can be changed from the static variable brainSize in Brain class:
```sh
brainSize = [4,5,6,4]
# 5 layers, input and output have 4 features, there are 2 hidden layers with 5 and 6 nodes respectively
# The best 'brain sizes' observed are [4,2,4] to [4,5,4] 
```
Every new generation the car that travelled the furthest gives its 'brain' to the next generation. One random mutation is performed to the biases and weights of each layer in the range of [0, 0.1].

#### Libraries used:
- [math]
- [pygame]
- [numpy]
- [time]
- [random]




### Future updates:
  - Spawn more cars with the next generation, some must be totaly random while the rest must be mutations of the previusly best performing brain.
  - Reduce the needed libraries.
  - Add a past position tracker and prevent the brain from choosing to move there, this might make the cars go around the maze, instead of going back and forth on the same 2-3 blocks.


### Installation

To run the code you need Python3, and the libraries above installed on your computer.
To install a libray for python open the command prompt and follow the example bellow.

```sh
$ pip install pygame
```

To clone the repository, open the command prompt at the directory of choice and type:
```sh
$  git clone --recursive https://github.com/rocku0/Simple-genetic-evolution-machine-learning-moving-blocks-through-a-maze
```

**Use this as you like**

   [math]: <https://docs.python.org/3/library/math.html>
   [pygame]: <https://www.pygame.org/docs/>
   [numpy]: <https://numpy.org/doc/>
   [time]: <https://docs.python.org/3/library/time.html>
   [random]: <https://docs.python.org/3/library/random.html>
