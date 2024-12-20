#!/usr/bin/env python3

from ultralytics import YOLO
import argparse
import cv2
import math
import numpy as np
from YOLO.model import tabulate_pieces

# train model: yolo task=detect mode=train epochs=20 data=C:\Users\lbrab\OneDrive\Documents\Programming\ComputerVisionFA23\SettlersEye\TileDetection\data\data.yaml model=yolov8m.pt imgsz=640

class Tile:
    def __init__(self, vertex1, vertex2):
        self.vertex1 = vertex1  # top left vertex
        self.vertex2 = vertex2  # bottom right vertex
        self.boxWidth = vertex2[0] - vertex1[0]
        self.boxHeight = vertex2[1] - vertex1[1]
        self.center = ((vertex2[0]+vertex1[0])//2, (vertex2[1]+vertex1[1])//2)
        self.area = (vertex2[0]-vertex1[0])*(vertex2[1]-vertex1[1])
        self.neighbors = set()
        self.vertices = []

    def __str__(self):
        return f"Tile(vertex1={self.vertex1}, vertex2={self.vertex2}, center={self.center}, area={self.area})"
    
class Graph:
    def __init__(self):
        pass

vertexNum = -1
class Vertex:
    def __init__(self, point):
        global vertexNum
        self.point = point
        self.id = vertexNum = vertexNum + 1

    def __str__(self):
        return f"Vertex(id={self.id}, point={self.point})"


def build_parser():
    parser = argparse.ArgumentParser(description="Python wrapper to run YOLOv8 model on images")
    parser.add_argument("--image", help="Image to classify", required=True)
    parser.add_argument("--image_size", help="Specify image size", required=False, default=640)
    parser.add_argument("--confidence", help="Specify image size", required=False, default=.90)
    parser.add_argument("--players", help="Specify players present (b for blue, r for red, o for orange, w for white) (ed: ow)", required=True)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    model = YOLO('yolov8m_tiledetection.pt')
    image = cv2.imread(args.image)

    # run tile detector
    tiles = []
    results = model.predict(args.image, imgsz=args.image_size, conf=args.confidence)[0]
    boxes = [(math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])) for box in results.boxes.xyxy.cpu().numpy()]
    if len(boxes) < 19:
        print("Error: failed to detect full catan board (detected fewer than 19 game tiles)\n")
        return
    for i in range(min(len(boxes), 19)): # detect at most 19 tiles
        tiles.append(Tile((boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3])))

    tiles.sort(key = lambda x: (x.center[1], x.center[0]))
    meanWidth = np.mean([tile.boxWidth for tile in tiles])
    for i in range(len(tiles)):
        for j in range(i+1, len(tiles)):
            if math.dist(tiles[i].center, tiles[j].center) <= 1.35*meanWidth: # add neighbors to list
                tiles[i].neighbors.add(j)
                tiles[j].neighbors.add(i)
    
    
    # deduce vertices of hexagonal tiles using geometrical properties and distance from neighbors
    image_w_tiles = image.copy()
    height, width, _= image_w_tiles.shape
    for i in range(len(tiles)):
        neighbor = tiles[list(tiles[i].neighbors)[0]]

        apothem = math.ceil(math.dist(tiles[i].center, neighbor.center))/2
        radius = int(apothem / 0.866025) # radius = apothem/cos(30)
        angleIncrement = 1.047197 # 2*PI/6
        angle = math.atan2(neighbor.center[1] - tiles[i].center[1], neighbor.center[0] - tiles[i].center[0])
        for _ in range(6):
            angle += angleIncrement
            tiles[i].vertices.append((int(tiles[i].center[0] + radius * math.sin(angle)), int(tiles[i].center[1] + radius * math.cos(angle))))

        # debug: draw hexagon
        pts = np.array(tiles[i].vertices, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_w_tiles, [pts], True, (36,255,12), height//500)
        # debug: write number of neighbors one tile
        cv2.putText(image_w_tiles, f"{len(tiles[i].neighbors)}", (tiles[i].center[0], tiles[i].center[1]), cv2.FONT_HERSHEY_SIMPLEX, height//1000, (36,255,12), height//1000)

    cv2.imshow("tile detection", cv2.resize(image_w_tiles, (500, int(height/(width/500)))))

    # merge vertices that are closer than 20% edge length
    for i in range(len(tiles)):
        for j in range(len(tiles)):
            for k in range(6):
                for l in range(6):
                    vertex1 = tiles[i].vertices[k]
                    vertex2 = tiles[j].vertices[l]
                    dist = int(math.dist(vertex1, vertex2))
                    if dist <= radius * .3:
                        tiles[j].vertices[l] = tiles[i].vertices[k]

    # verify that board consists of exactly 54 vertices
    vertices = set()
    for i in range(len(tiles)): 
        for vertex in tiles[i].vertices:
            vertices.add(vertex)
    if (len(vertices) != 54):
        print(f"Error: failed to detect full catan board ({len(vertices)} vertices detected when 54 expected)\n")
        # for v in vertices:
        #     print(v)
        return
    
    # create graph where edges are tile edges and vertices are tile vertices (used for validating placement of pieces)
    board_E = []
    board_V = list(vertices)
    for i in range(len(board_V)):
        for j in range(i+1, len(board_V)):
            if math.dist(board_V[i], board_V[j]) <= radius * 1.25:
                board_E.append((board_V[i], board_V[j]))

    # create graph where vertices are centerpoints of tile edges (used for finding longest road)
    roads_V = []
    image_w_roads = image.copy()
    height, width, _= image_w_roads.shape
    for tileEdge in board_E:
        roads_V.append(((tileEdge[0][0]+tileEdge[1][0])//2, (tileEdge[0][1]+tileEdge[1][1])//2))

    roads_Adjacency = dict()
    for i in range(len(roads_V)):
        for j in range(i+1, len(roads_V)):
            if math.dist(roads_V[i], roads_V[j]) <= radius * 1.25:
                if roads_V[i] not in roads_Adjacency:
                    roads_Adjacency[roads_V[i]] = []
                roads_Adjacency[roads_V[i]].append(roads_V[j])

                if roads_V[j] not in roads_Adjacency:
                    roads_Adjacency[roads_V[j]] = []
                roads_Adjacency[roads_V[j]].append(roads_V[i])
    
    for v1 in roads_Adjacency:
        for v2 in roads_Adjacency[v1]:
            image_w_roads = cv2.line(image_w_roads, v1, v2, (0,255,0), height//500)

    # detect roads and pieces
    model = YOLO("med_model.pt")
    results = model.predict(args.image, imgsz=args.image_size, conf=.4)[0]
    scores = tabulate_pieces(results)

    player_points = {}
    longest = 0
    longest_road_recipients = []
    if "b" in args.players:
        points, longest_path, detected_roads = score_player(scores["blue"], board_V, roads_V, roads_Adjacency, radius)
        for road in detected_roads:
            image_w_roads = cv2.circle(image_w_roads, road, radius=0, color=(255, 0, 0), thickness=height//50)
        player_points["blue"] = points
        if longest_path == longest:
            longest_road_recipients.append("blue")
        elif longest_path > longest:
            longest_road_recipients = ["blue"]
            longest = longest_path

    if "r" in args.players:
        points, longest_path, detected_roads = score_player(scores["red"], board_V, roads_V, roads_Adjacency, radius)
        for road in detected_roads:
            image_w_roads = cv2.circle(image_w_roads, road, radius=0, color=(0, 0, 255), thickness=height//50)
        player_points["red"] = points
        if longest_path == longest:
            longest_road_recipients.append("red")
        elif longest_path > longest:
            longest_road_recipients = ["red"]
            longest = longest_path

    if "o" in args.players:
        points, longest_path, detected_roads = score_player(scores["orange"], board_V, roads_V, roads_Adjacency, radius)
        for road in detected_roads:
            image_w_roads = cv2.circle(image_w_roads, road, radius=0, color=(0, 165, 255), thickness=height//50)
        player_points["orange"] = points
        if longest_path == longest:
            longest_road_recipients.append("orange")
        elif longest_path > longest:
            longest_road_recipients = ["orange"]
            longest = longest_path

    if "w" in args.players:
        points, longest_path, detected_roads = score_player(scores["white"], board_V, roads_V, roads_Adjacency, radius)
        for road in detected_roads:
            image_w_roads = cv2.circle(image_w_roads, road, radius=0, color=(255, 255, 255), thickness=height//50)
        player_points["white"] = points
        if longest_path == longest:
            longest_road_recipients.append("white")
        elif longest_path > longest:
            longest_road_recipients = ["white"]
            longest = longest_path

    print("\n#################################### SETTLERS EYE ####################################")
    print("# LONGEST ROAD:")
    if len(longest_road_recipients) == 1:
        player_points[longest_road_recipients[0]] += 2
        print(f"#  - Longest road was awarded to {longest_road_recipients[0]}.")
        print(f"#  - The points below reflect {longest_road_recipients[0]} receiving longest road.")
    elif len(longest_road_recipients) > 1:
        print(f"#  - The following players tied for longest road: {', '.join(longest_road_recipients)}.")
        print(f"#  - 2 additional points should be awarded to the player who reached length {longest} road path first.")
    print("# POINT SUMMARY:")
    for color in player_points:
        print(f"#  - {color}: {player_points[color]} points")
    print("# LARGEST ARMY:")
    print("#  - Award two additional points to the player with the largest army")
    print("######################################################################################")


    cv2.imshow("road graph", cv2.resize(image_w_roads, (500, int(height/(width/500)))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def score_player(detections, board_V, roads_V, roads_Adjacency, radius):
    points = 0
    for settlement in detections["settlement"]:
        for v in board_V:
            if math.dist(settlement, v) < radius * .4:
                points += 1
                break
    for city in detections["city"]:
        for v in board_V:
            if math.dist(city, v) < radius * .4:
                points += 2
                break

    detected_roads = set()
    for road in detections["road"]:
        for v in roads_V:
            if math.dist(road, v) < radius * .4:
                detected_roads.add(v)
                break
    
    longest_path = 0
    for road in detected_roads:
        path_len = longest_road(road, roads_Adjacency, detected_roads, 1, {road})
        if path_len > longest_path:
            longest_path = path_len
    
    return points, longest_path, detected_roads


def longest_road(curr, roads_Adjacency, detected_roads, length, visited):
    longest = length
    for road in roads_Adjacency[curr]:
        if road in visited:
            continue
        if road in detected_roads:
            new_visited = visited.copy()
            new_visited.add(road)
            path_len = longest_road(road, roads_Adjacency, detected_roads, length+1, new_visited)
            if path_len > longest:
                longest = path_len
    
    return longest



if __name__ == "__main__":
    main()
