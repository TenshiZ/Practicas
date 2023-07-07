import sys

class Node():
    def __init__(self,state,parent,action):
        self.state  = state
        self.parent = parent
        self.action = action

class StackFrontier():
    def __init__(self):
        self.frontier=[]
    
    def add(self,node):
        self.frontier.append(node)
    
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)
    
    def empty(self):
        return len(self.frontier) == 0
    
    def remove(self):
        if(self.empty()):
            raise Exception("Empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node
            

class QueueFrontier(StackFrontier):
    
    def remove(self):
        if(self.empty()):
            raise Exception("Empty frontier")
        else:
            node = self.frontier[0]
            self.frontier = self.frontier[1:]
            return node
        
class Maze():
    def __init__(self,filename):
        with open(filename) as f:
            contents = f.read()
        
        if contents.count("A") != 1:
            raise Exception("maze must have exactly one start point")
        if contents.count("B") != 1:
            raise Exception("maze must have exactly one goal")
        
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        self.walls =[]

        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == "A":
                        self.start = (i,j)
                        row.append(False)
                    elif contents[i][j] == "B":
                        self.goal = (i,j)
                        row.append(False)
                    elif contents[i][j] == " ":
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)
        self.solution = None
    
    def print(self):
        solution = self.solution[1] if self.solution is not None else None
        print()
        for i,row in enumerate(self.walls):
            for j,col in enumerate(row):
                if col :
                    print("■", end="")
                elif (i,j) == self.start:
                    print("A", end="")
                elif (i,j) == self.goal:
                    print("B", end="")
                elif solution is not None and (i,j) in solution:
                    print("*", end="")
                else:
                    print(" ",end="")
            print()
        print()
    
    def neightbors(self,state):
        row,col = state
        
        candidates = [
            ("up", (row - 1,col)),
            ("down", (row + 1,col)),
            ("left", (row,col - 1)),
            ("right", (row,col + 1))
        ]

        result = []
        for action, (r,c) in candidates:
            try:
                if not self.walls[r][c]:
                    result.append((action, (r,c)))
            except IndexError:
                continue
        return result
    
    def solve(self):
        
        self.num_explored = 0

        start = Node(state = self.start, parent=None,action=None)
        frontier = StackFrontier()
        frontier.add(start)

        self.explored = set()

        while True:
            if frontier.empty():
                raise Exception("No solution")
            
            node = frontier.remove()
            self.num_explored += 1

            if node.state == self.goal:
                actions = []
                cells = []

                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return
            
            self.explored.add(node.state)

            for action, state in self.neightbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state = state, parent = node,action = action)
                    frontier.add(child)
    
    def output_image(self, filename, show_solution = True, show_explored = False):
        from PIL import Image, ImageDraw

        cell_size = 50
        cell_border = 2

        # Create a blank image with white background
        img_width = self.width * cell_size
        img_height = self.height * cell_size
        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):
                # Calculate the coordinates of the cell in the image
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                if self.walls[i][j]:
                    # Draw a black rectangle for walls
                    draw.rectangle((x1, y1, x2, y2), fill="black")
                elif (i, j) == self.start:
                    # Draw a green rectangle for the start position
                    draw.rectangle((x1, y1, x2, y2), fill="green")
                elif (i, j) == self.goal:
                    # Draw a red rectangle for the goal position
                    draw.rectangle((x1, y1, x2, y2), fill="red")
                elif show_solution and self.solution is not None and (i, j) in self.solution[1]:
                    # Draw a blue rectangle for cells in the solution path
                    draw.rectangle((x1, y1, x2, y2), fill="blue")
                elif show_explored and (i, j) in self.explored:
                    # Draw a light gray rectangle for explored cells
                    draw.rectangle((x1, y1, x2, y2), fill="lightgray")

        # Save the image to a file
        img.save(filename)



if len(sys.argv)!= 2:
    sys.exit("Usage: python3 maze.py maze.txt")

m = Maze(sys.argv[1])
print("Maze:")
m.print()
print("Solving...")
m.solve()

print("States Explored:", m.num_explored)
print("Solution:")
m.print()
m.output_image("maze.png", show_explored = True)

