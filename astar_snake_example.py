from collections import deque
from queue import PriorityQueue
import random
import tkinter as tk
from typing import Tuple, List
import copy
#---------以下为可改参数-------------
WIDTH = 200  # 窗口宽度
HEIGHT = 200  # 窗口高度
SIZE = 20  # 格子大小，一定要整除
FPS = 100  # 刷新率，填写范围为0-1000,数字越大移动越快
#-------------------------------------

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
#请不要修改
class Snake:
    def __init__(self) -> None:
        self.body: List[Tuple[int, int]] = [
            (5, 5), (5, 6)]
        self.direction = RIGHT
        self.path = []

    def head(self) -> Tuple[int, int]:
        return self.body[0]

    def tail(self) -> Tuple[int, int]:
        return self.body[-1]

    def move(self, food):
        if self.direction == UP:
            if (self.head()[0], self.head()[1]-1) == food:
                self.body.insert(0, (self.head()[0], self.head()[1]-1))
            else:
                self.body.insert(0, (self.head()[0], self.head()[1]-1))
                self.body.pop()
        elif self.direction == DOWN:
            if (self.head()[0], self.head()[1]+1) == food:
                self.body.insert(0, (self.head()[0], self.head()[1]+1))
            else:
                self.body.insert(0, (self.head()[0], self.head()[1]+1))
                self.body.pop()
        elif self.direction == LEFT:
            if (self.head()[0]-1, self.head()[1]) == food:
                self.body.insert(0, (self.head()[0]-1, self.head()[1]))
            else:
                self.body.insert(0, (self.head()[0]-1, self.head()[1]))
                self.body.pop()
        elif self.direction == RIGHT:
            if (self.head()[0]+1, self.head()[1]) == food:
                self.body.insert(0, (self.head()[0]+1, self.head()[1]))
            else:
                self.body.insert(0, (self.head()[0]+1, self.head()[1]))
                self.body.pop()

    def change_direction(self, direction) -> None:
        if direction == UP and self.direction != DOWN:
            self.direction = UP
        elif direction == DOWN and self.direction != UP:
            self.direction = DOWN
        elif direction == LEFT and self.direction != RIGHT:
            self.direction = LEFT
        elif direction == RIGHT and self.direction != LEFT:
            self.direction = RIGHT

    def is_dead(self) -> bool:
        if self.head()[0] < 0 or self.head()[0] >= WIDTH // SIZE or self.head()[1] < 0 or self.head()[1] >= HEIGHT//SIZE:
            return True
        for i in range(1, len(self.body)):
            if self.head() == self.body[i]:
                return True
        return False

    def reset(self):
        self.body = [(5, 5), (5, 6)]
        self.direction = RIGHT


class Food:
    def __init__(self, snake) -> None:
        self.position = (10, 10)
        self.new_food(snake)

    def new_food(self, snake: Snake):
        """创建新食物，如果新食物在蛇身内，则生成新事物"""
        self.position = (random.randint(0, WIDTH//SIZE - 1),
                         random.randint(0, HEIGHT//SIZE - 1))
        for node in snake.body:
            if self.position == node:
                self.new_food(snake)
            if len(snake.body) >= (WIDTH//SIZE)*(HEIGHT//SIZE):
                break


class Game:
    def __init__(self) -> None:
        self.snake = Snake()
        self.food = Food(self.snake)
        self.score = 0

    def is_over(self):
        return self.snake.is_dead()

    def update(self):
        self.snake.move(self.food.position)
        if self.food.position == self.snake.head():
            self.score += 1
            self.food.new_food(self.snake)
            self.snake.path.clear()

    def change_direction(self, direction):
        self.snake.change_direction(direction)

    def reset(self):
        self.score = 0
        self.food.new_food(self.snake)
        self.snake.reset()

    def draw_snake(self, canvas: tk.Canvas):
        for (x, y) in self.snake.body:
            if (x, y) == self.snake.head():
                canvas.create_rectangle(
                    x*SIZE, y*SIZE, (x+1)*SIZE, (y+1)*SIZE, fill="blue")
            elif (x, y) == self.snake.tail():
                canvas.create_rectangle(
                    x*SIZE, y*SIZE, (x+1)*SIZE, (y+1)*SIZE, fill="green")
            else:
                canvas.create_rectangle(
                    x*SIZE, y*SIZE, (x+1)*SIZE, (y+1)*SIZE, fill="black")

    def draw_food(self, canvas: tk.Canvas):
        x, y = self.food.position
        canvas.create_oval(x*SIZE, y*SIZE, (x+1)*SIZE, (y+1)*SIZE, fill="red")

    def draw_score(self, canvas: tk.Canvas):
        canvas.create_text(
            10, 10, text=f"score:{self.score}", anchor="nw", fill="green")

    def AI_move(self):
        """AI思考模块"""
        if self.snake.path:  # 这里是指虚拟蛇已经提供了路径，就不需要再重复计算了，直接套用虚拟蛇的路径
            self.change_direction(self.snake.path[1])
            self.snake.path.pop(0)
            return
        gh = Graph(self.snake)
        path = a_star_search(gh, self.snake.head(), self.food.position)
        neighbors = gh.neighbors(self.snake.head())
        if self.food.position in path:  # 当有这条路径时，虚拟蛇走，看走之后安不安全,如果安全就走吧，如果不安全，就去追尾巴
            new_game = copy.deepcopy(self)
            safe = None
            while new_game.food.position != new_game.snake.head():
                new_gh = Graph(new_game.snake)
                new_path = a_star_search(
                    new_gh, new_game.snake.head(), new_game.food.position)
                virtual_path = turn_of_next(new_path[1], new_game)
                new_game.snake.path.append(virtual_path)
                new_game.snake.move(new_game.food.position)
                if new_game.food.position == new_game.snake.head() and is_safe(new_game, new_game.snake.head()):
                    safe = True
                    self.snake.path = new_game.snake.path
                    break

            if safe:
                ddd = turn_of_next(path[1], self)
            else:  # 追尾巴
                ddd = turn_of_next(bfs_path_search(
                    neighbors, gh.matrix, self.snake.tail()), self)
        else:
            ddd = turn_of_next(bfs_path_search(
                neighbors, gh.matrix, self.snake.tail()), self)
        if ddd == None:
            if self.snake.tail() in get_all_neighbors(self.snake.head(), gh.matrix):
                ddd = turn_of_next(self.snake.tail(), self)
            else:
                if neighbors:
                    ddd = turn_of_next(random.choice(neighbors), self)
                else:
                    print("得分为：",self.score,"总分为：",(WIDTH//SIZE)*(HEIGHT//SIZE))


class window():
    def __init__(self) -> None:
        root = tk.Tk()
        root.title("贪吃蛇 -by 谢鹏")
        self.bg_color = "white"
        self.screenwidth = root.winfo_screenwidth()
        self.screenheight = root.winfo_screenheight()
        self.size_geo = '%dx%d+%d+%d' % (WIDTH, HEIGHT,
                                         (self.screenwidth-WIDTH)/2, (self.screenheight-HEIGHT)/2)
        root.geometry(self.size_geo)
        btn1 = tk.Button(root, text="开始游戏", command=self.start_play)
        btn2 = tk.Button(root, text="看AI玩", command=self.AI_play)
        btn4 = tk.Button(root, text="退出游戏", command=root.destroy)
        btn1.grid(row=0, column=0, padx=10, pady=10)
        btn2.grid(row=0, column=1, padx=10, pady=10)
        btn4.grid(row=0, column=2, padx=10, pady=10)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(0, weight=1)
        root.mainloop()

    def start_play(self):
        """单人游玩模块"""
        map = tk.Tk()
        map.title("单人游玩,按q退出")
        map.geometry(self.size_geo)
        canvas = tk.Canvas(map, width=WIDTH, height=HEIGHT, bg="white")
        canvas.pack()
        game = Game()

        def key_press(event):
            if event.keysym == "Up" or event.keysym == "w":
                game.change_direction(UP)
            elif event.keysym == "Down" or event.keysym == "s":
                game.change_direction(DOWN)
            elif event.keysym == "Left" or event.keysym == "a":
                game.change_direction(LEFT)
            elif event.keysym == "Right" or event.keysym == "d":
                game.change_direction(RIGHT)
            elif event.keysym == "q":
                map.destroy()
        map.bind("<KeyPress>", key_press)

        def update():
            if game.is_over():
                game.reset()
            canvas.delete("all")
            game.update()
            game.draw_food(canvas)
            game.draw_snake(canvas)
            game.draw_score(canvas)
            map.after(1000//FPS, update)
        update()
        map.mainloop()

    def AI_play(self):
        """AI游玩界面"""
        map = tk.Tk()
        map.title("AI游玩")
        map.geometry(self.size_geo)
        canvas = tk.Canvas(map, width=WIDTH, height=HEIGHT, bg="white")
        canvas.pack()
        game = Game()

        def update():
            if game.is_over():
                print("得分为：", game.score,"总分为：",(WIDTH//SIZE)*(HEIGHT//SIZE))
                if len(game.snake.body) >= (WIDTH//SIZE)*(HEIGHT//SIZE):
                    print("你赢了!")
                game.reset()
            canvas.delete("all")
            game.AI_move()
            game.update()
            game.draw_food(canvas)
            game.draw_snake(canvas)
            game.draw_score(canvas)
            map.after(1000//FPS, update)
        update()
        map.mainloop()


class Graph:
    """地图信息"""

    def __init__(self, snake: Snake) -> None:
        self.snake = copy.deepcopy(snake)
        self.matrix = [[0 if (j, i) not in snake.body else 1 for j in range(
            WIDTH//SIZE)] for i in range(HEIGHT//SIZE)]

    def neighbors(self, current: tuple) -> list:
        """某个点附近的可走的点"""
        x = current[0]
        y = current[1]
        empty = []
        if x-1 >= 0 and y >= 0 and x-1 < WIDTH//SIZE and y < HEIGHT//SIZE and ((x-1, y) not in self.snake.body):
            empty.append((x-1, y))
        if x+1 >= 0 and y >= 0 and x+1 < WIDTH//SIZE and y < HEIGHT//SIZE and ((x+1, y) not in self.snake.body):
            empty.append((x+1, y))
        if x >= 0 and y-1 >= 0 and x < WIDTH//SIZE and y-1 < HEIGHT//SIZE and ((x, y-1) not in self.snake.body):
            empty.append((x, y-1))
        if x >= 0 and y+1 >= 0 and x < WIDTH//SIZE and y+1 < HEIGHT//SIZE and ((x, y+1) not in self.snake.body):
            empty.append((x, y+1))
        return empty


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1-x2)+abs(y1-y2)


def a_star_search(graph: Graph, start, goal):
    """搜索地图上从起点到终点的最短路径
    graph:Graph
    start:开始
    goal:终点"""
    frontier = PriorityQueue()
    frontier.put(start, False)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current]+1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost+heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    current = goal
    path = []
    if current in came_from:
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)  # optional
        path.reverse()  # optional
        return path
    else:
        return [start, (-1, -1)]


def bfs(start, matrix):
    visited = [[False for _ in row] for row in matrix]
    distance_matrix = [[-1 for _ in row] for row in matrix]
    x, y = start
    visited[x][y] = True
    distance_matrix[x][y] = 0

    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[0]) and matrix[nx][ny] != 1 and not visited[nx][ny]:
                visited[nx][ny] = True
                distance_matrix[nx][ny] = distance_matrix[x][y] + 1
                queue.append((nx, ny))

    return distance_matrix


def bfs_path_search(neighbors, matrix, start):
    distance_matrix = bfs((start[1], start[0]), matrix)
    max = -1
    max_i = -1
    max_j = -1
    for (i, j) in neighbors:
        if distance_matrix[j][i] > max:
            max_i = i
            max_j = j
            max = distance_matrix[j][i]
    return (max_i, max_j)


def turn_of_next(next_path: tuple, game: Game):
    """根据坐标转向"""
    x = next_path[0]
    y = next_path[1]
    if x == game.snake.head()[0]-1 and y == game.snake.head()[1]:
        game.change_direction(LEFT)
        return LEFT
        #print("turn left")
    elif x == game.snake.head()[0]+1 and y == game.snake.head()[1]:
        game.change_direction(RIGHT)
        return RIGHT
        #print("turn right")
    elif y == game.snake.head()[1]-1 and x == game.snake.head()[0]:
        game.change_direction(UP)
        return UP
        #print("turn up")
    elif y == game.snake.head()[1]+1 and x == game.snake.head()[0]:
        game.change_direction(DOWN)
        return DOWN
        #print("turn down")


def get_all_neighbors(point, matrix):
    """返回一个点四周四个点的列表"""
    x, y = point
    height = len(matrix)
    width = len(matrix[0])
    candidates = [
        (x+1, y),
        (x-1, y),
        (x, y+1),
        (x, y-1)
    ]
    neighbors = []
    for nx, ny in candidates:
        if 0 <= nx < height and 0 <= ny < width and matrix[nx][ny] != -2:
            neighbors.append((nx, ny))
    return neighbors


def is_safe(game: Game, start):
    """根据游戏当前形态判断是否有通往蛇尾的路径"""
    new_gh = Graph(game.snake)
    new_gh.snake.body.pop()
    tail = game.snake.tail()
    path_to_tail = a_star_search(new_gh, start, tail)
    if tail in path_to_tail:
        return path_to_tail
    else:
        return None


if __name__ == "__main__":
    wd = window()