import random
import math
import timeit
import matplotlib.pyplot as plt


# point class with x, y as point
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def as_tuple(self):
        return (self.x, self.y)


class TestDataGenerator:
    """
    A class to represent a synthetic data generator.

    ...

    Attributes
    ----------

    [to be defined as part of the coursework]

    Methods
    -------

    [to be defined as part of the coursework]

    """

    # ADD YOUR CODE HERE
    def __init__(self, num_points=100):
        self.num_points = num_points

    def generate_random_points(self) -> list[Point]:
        return [
            Point(random.randrange(0, 100), random.randrange(0, 100))
            for _ in range(self.num_points)
        ]

    def generate_spiral_points(self) -> list[Point]:
        points: list[Point] = []
        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            x = math.cos(angle) * i
            y = math.sin(angle) * i
            points.append(Point(x, y))
        return points

    def generate_collinear_x_points(self) -> list[Point]:
        points = [Point(i, 0) for i in range(self.num_points)]
        return points

    def generate_circle_points(self) -> list[Point]:
        points: list[Point] = []
        for i in range(self.num_points):
            angle = 2 * math.pi * i / self.num_points
            x = math.cos(angle)
            y = math.sin(angle)
            points.append(Point(x, y))
        return points


def average(arr: list):
    return sum(arr) / len(arr)


class ExperimentalFramework:
    """
    A class to represent an experimental framework.

    ...

    Attributes
    ----------

    [to be defined as part of the coursework]

    Methods
    -------

    [to be defined as part of the coursework]

    """

    def __init__(self, num_points: int = 100):
        self.num_points = num_points
        self.dataGen = TestDataGenerator(num_points)
        # self.test_random()
        # self.test_spiral()
        # self.test_collinear()

    def test(self, testing_func, funcs):
        funcs = [{"name": f.__name__, "func": f} for f in funcs]
        results = [
            {"name": func["name"], "score": testing_func(func["func"])}
            for func in funcs
        ]
        fastest = min(results, key=lambda x: x["score"])
        fastest_name = fastest["name"]
        print(f"\x1b[1;32m{fastest_name.upper()} IS THE FASTEST\x1b[0m")
        slowest = max(results, key=lambda x: x["score"])
        slowest_name = slowest["name"]
        print(f"\x1b[1;31m{slowest_name.upper()} IS THE SLOWEST\x1b[0m")

    def test_random(self, funcs):
        self.test(self.test_random_points, funcs)

    def test_spiral(self, funcs):
        self.test(self.test_spiral_points, funcs)

    def test_collinear(self, funcs):
        self.test(self.test_collinear_points, funcs)

    def test_random_points(self, test_func) -> float:
        return self.test_function(
            test_func,
            TestDataGenerator.generate_random_points,
            "\x1b[1;34mRANDOM POINTS",
        )

    # worst case for jarvis march
    def test_spiral_points(self, test_func) -> float:
        return self.test_function(
            test_func,
            TestDataGenerator.generate_spiral_points,
            "\x1b[1;35mSPIRAL POINTS",
        )

    # worst case for grahamscan
    def test_collinear_points(self, test_func) -> float:
        return self.test_function(
            test_func,
            TestDataGenerator.generate_collinear_x_points,
            "\x1b[1;33mCOLLINEAR POINTS",
        )

    def test_function(
        self,
        test_func,
        gen_func,
        title: str = "\x1b[1;34mRANDOM POINTS",
    ) -> float:
        test_func_name = test_func.__name__
        STMT = f"stmt=TestDataGenerator({self.num_points}).{gen_func.__name__}();{test_func_name}(stmt)"
        SETUP = f"from __main__ import TestDataGenerator, {test_func_name}"
        print(
            f"\x1b[1;36mTESTING {test_func_name.upper()} - {self.num_points} {title.strip()} \x1b[0m"
        )
        toc = average(timeit.repeat(stmt=STMT, setup=SETUP, number=1, repeat=2))
        print(f"\x1b[32mTime taken:\x1b[0m {toc}")
        return toc


# Python3 program to find convex hull of a set of points. Refer
# https://www.geeksforgeeks.org/orientation-3-ordered-points/
# for explanation of orientation()


def left_index(points: list[Point]):
    """
    Finding the left most point
    """
    minn = 0
    for i in range(1, len(points)):
        if points[i].x < points[minn].x:
            minn = i
        elif points[i].x == points[minn].x:
            if points[i].y > points[minn].y:
                minn = i
    return minn


def orientation(p: Point, q: Point, r: Point):
    """
    To find orientation of ordered triplet (p, q, r).
    The function returns following values
    0 --> p, q and r are collinear
    1 --> Clockwise
    2 --> Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)

    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2


def jarvismarchConvexHull(points: list[Point], n: int):

    # There must be at least 3 points
    if n < 3:
        return

    # Find the leftmost point
    ligma_balls = left_index(points)

    hull = []

    """ 
Start from leftmost point, keep moving counterclockwise 
until reach the start point again. This loop runs O(h) 
times where h is number of points in result or output. 
"""
    p = ligma_balls
    q = 0
    while True:

        # Add current point to result
        hull.append(p)

        """ 
Search for a point 'q' such that orientation(p, q, 
x) is counterclockwise for all points 'x'. The idea 
is to keep track of last visited most counterclock- 
wise point in q. If any point 'i' is more counterclock- 
wise than q, then update q. 
"""
        q = (p + 1) % n

        for i in range(n):

            # If i is more counterclockwise
            # than current q, then update q
            if orientation(points[p], points[i], points[q]) == 2:
                q = i

        """ 
Now q is the most counterclockwise with respect to p 
Set p as q for next iteration, so that q is added to 
result 'hull' 
"""
        p = q

        # While we don't come to first point
        if p == ligma_balls:
            break


def jarvismarch(points: list[Point]):
    return jarvismarchConvexHull(points, len(points))


# Function to compute the cross product of three points
def cross_product(p1: Point, p2: Point, p3: Point) -> float:
    return (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)


# Function to determine if the turn is left
def left_turn(p1: Point, p2: Point, p3: Point) -> bool:
    return cross_product(p1, p2, p3) > 0


# Function to compute the upper hull of a set of points
def upper_hull(points: list[Point]) -> list[Point]:
    upper = []
    for p in points:
        while len(upper) >= 2 and not left_turn(upper[-2], upper[-1], p):
            upper.pop()
        upper.append(p)
    return upper


# Function to compute the lower hull of a set of points
def lower_hull(points: list[Point]) -> list[Point]:
    lower = []
    for p in reversed(points):
        while len(lower) >= 2 and not left_turn(lower[-2], lower[-1], p):
            lower.pop()
        lower.append(p)
    return lower


# Function to compute the convex hull using Chan's algorithm
def chans(points: list[Point]) -> list[Point]:
    # Sort points by x-coordinate
    points.sort(key=lambda p: (p.x, p.y))

    # Split points into blocks of size B
    B = int(math.sqrt(len(points)))
    blocks = [points[i : i + B] for i in range(0, len(points), B)]

    # Compute upper hull of each block
    upper_hulls = [upper_hull(block) for block in blocks]

    # Merge upper hulls
    upper = []
    for hull in upper_hulls:
        while len(upper) >= 2 and not left_turn(upper[-2], upper[-1], hull[0]):
            upper.pop()
        upper.extend(hull)

    # Compute lower hull of each block
    lower_hulls = [lower_hull(block) for block in blocks]

    # Merge lower hulls
    lower = []
    for hull in reversed(lower_hulls):
        while len(lower) >= 2 and not left_turn(lower[-2], lower[-1], hull[0]):
            lower.pop()
        lower.extend(hull)

    # Remove duplicates and return convex hull
    return list(set(upper + lower))


def quick_sort(arr, key=lambda n: n):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if key(x) < key(pivot)]
    middle = [x for x in arr if key(x) == key(pivot)]
    right = [x for x in arr if key(x) > key(pivot)]
    return quick_sort(left, key) + middle + quick_sort(right, key)


def merge_sort(arr, key=lambda n: n):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid], key)
    right = merge_sort(arr[mid:], key)

    return merge(left, right, key)


def merge(left, right, key=lambda n: n):
    result = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if key(left[left_idx]) < key(right[right_idx]):
            result.append(left[left_idx])
            left_idx += 1
        else:
            result.append(right[right_idx])
            right_idx += 1

    result.extend(left[left_idx:])
    result.extend(right[right_idx:])
    return result


def heapify(arr, n, i, key=lambda x: x):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and key(arr[left]) > key(arr[largest]):
        largest = left

    if right < n and key(arr[right]) > key(arr[largest]):
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, key)


def heap_sort(arr, key=lambda x: x):
    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, key)

    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0, key)


def grahamscan(inputSet: list[Point], sort_func=heap_sort):
    """
    Returns the list of points that lie on the convex hull (Graham Scan algorithm)
            Parameters:
                    inputSet (list): a list of 2D points

            Returns:
                    outputSet (list): a list of 2D points
    """
    inputSet = [p.as_tuple() for p in inputSet]

    # Find the orientation of three points (p, q, r)
    # 0 -> Colinear
    # 1 -> Clockwise
    # 2 -> Counterclockwise
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Colinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    # Function to find the next-to-top element in the stack
    def next_to_top(S):
        return S[-2]

    from math import atan2  # Import atan2 function

    n = len(inputSet)
    if n < 3:
        return []

    # Find the bottommost point
    ymin = min(inputSet, key=lambda point: point[1])
    ymin_idx = inputSet.index(ymin)

    # Swap the bottommost point with the first point
    inputSet[0], inputSet[ymin_idx] = inputSet[ymin_idx], inputSet[0]

    # Sort points by polar angle with respect to the bottommost point
    def polar_angle(p):
        return atan2(p[1] - inputSet[0][1], p[0] - inputSet[0][0])

    inputSet[1:] = sort_func(inputSet[1:], key=polar_angle)

    # Initialize stack
    stack = [inputSet[0], inputSet[1], inputSet[2]]

    for i in range(3, n):
        # Keep removing the top while the angle formed by points next-to-top, top, and points[i] makes a non-left turn
        while (
            len(stack) > 1
            and orientation(next_to_top(stack), stack[-1], inputSet[i]) != 2
        ):
            stack.pop()
        stack.append(inputSet[i])

    outputSet = stack
    return outputSet


def main():

    ypoints = []

    MAX_POINTS = 1_000_000
    STEP = 20_000
    inputs = list(range(STEP, MAX_POINTS, STEP))
    plt.title("Testing grahamscan (Merge sort) - Collinear points (Worst case)")
    plt.xlabel("Number of points")
    plt.ylabel("Average compute time")

    for inp in inputs:
        e = ExperimentalFramework(inp)
        t = e.test_function(
            grahamscan,
            TestDataGenerator.generate_collinear_x_points,
            "COLLINEAR X POINTS",
        )
        ypoints.append(t)

    plt.plot(inputs[: len(ypoints)], ypoints)
    plt.show()


if __name__ == "__main__":
    main()
