# SettlersEye
This is a semester long project for Computer Vision (CSE 40535) that automatically scores a Settlers of Catan game provided an image of the board.

## Part 1: Conceptual Design
For our computer vision class project, we will be creating a system to look at a Settlers of Catan board (either from a finished game or one in progress) and calculate the current score for each player. Our project will calculate each player's points based on how many settlements and cities they have and who has the longest road. Additional victory points and the size of each players’ army will be inputted programmatically. 

Taking into account the fact that we must build the dataset ourselves, due to a lack of highly specific board game computer vision datasets (shocking), we must use techniques that accommodate for this limitation. Because of this, there are two general methods we hope to try. One of us will use traditional computer vision methods like edge detection and feature extraction and the other will use few-shot/one-shot learning. The features that we need to be able to recognize include simple shape, color, position, and orientation which should be compatible for the two solutions we will be pursuing.  

For the first solution (traditional techniques), we will use something like opencv for edge detection, feature extraction, and color classification. After detecting all of the pieces (roads, settlements, cities, and game tiles), their colors, and their positions, we will need to map each piece to a space, most likely some type of grid, that will allow us to see the bigger picture of how certain pieces are connected. This will matter primarily for game tiles and roads. Knowing where each game tile is will allow us to create an image of game tile edges which we can use to determine which player has the longest road, an achievement that is worth 2 victory points. Creating an image of game tile edges will also be necessary for the second solution so it is likely that aspects of this first solution, regardless of which is selected for detecting colors and settlements vs cities, will be used in our final project submission.  

After doing some research into the few-shot learning method, it seems like we can use a siamese network and contrastive loss to classify the piece type and color which will allow us to reconstruct the state of the board from the several matching steps we will be performing. Again, we will need to map the identified pieces to a usable grid so that we can calculate important features like longest road.  

When it comes to the dataset, the starting point will look different for the two techniques. For the training data, it will include pictures of valid game boards with the different pieces labeled, probably including some different angles. The verification data will probably be some similar images just to make sure that the techniques are working. For test data, the images will probably be similar to the training data but with different conditions such as different angles or lighting. For the FSL technique, the starting point should be images of each piece so the siamese network can correctly identify each object detected in the image. The verification data should be a mix of both board states and images of pieces to make sure the whole system works as well as the siamese network is functioning as well. For test data, it should look pretty similar to the traditional method’s test data. To create our training and verification data, we will use something similar to Labellmg (final tool to be chosen based on instructor’s suggestions).  

### Contributions of Each Team Member:
- Matthew Carbonaro: few-shot learning and dataset creation
- Luke Braby: introduction paragraphs and traditional techniques

## Part 2: Datasets
For this step of the process, we each captured 12 pictures of Catan game board and labeled each piece within the image (settlements, cities, roads, and tiles). 
- Source: our iPhone cameras
- Training vs Validation Datasets: The training dataset images do not necessarly depict a valid game state and often have more pieces than an average game would. The validation dataset consists of images of typical game states.
- Number of Objects: each image has 19 tiles, and roughly 14 cities, 16 settlements, and 54 roads. 
- Characteristics of Samples: Our images are taken from different angles and lightings to make the end model more general.
### Contributions of Each Team Member:
- Matthew Carbonaro: 12 photos with labels and collaborated on report
- Luke Braby: 12 photos with labels and collaborated on report

## Part 3: Preprocessing and Feature Extraction
For this step of the project, we tried traditional feature extraction methods and eventually decided to use trained models for feature extraction.
### Attempt at Traditional FEX Techniques:
Going into this phase of the project, we tried a few different methods of feature extraction. This section is dedicated to outlining the methods we tried and pointing out what may be helpful for future steps.  

The first feature we tried to extract was color. Excluding tiles, there are four colors that appear on the gameboard, with each color signifying who the piece belongs to. Because of this, being able to identify the color of each piece is an important capability to have. We were able to distinguish between blue, red, orange, and white pieces using hue values to differentiate between the three first colors and a greyscale threshold to identify white pieces. A collab notebook with this work can be found [here](https://colab.research.google.com/drive/1N_m_nfRODxJ9ASkbFAUyqjsePiQX6y4Y#scrollTo=QRLYJcjhjCmD). The code was not added to the git repo since one of our YOLO models is able to identify the colors of the pieces it detects.  

For identifying tiles, we tried using a SIFT descriptors to detect each tile, but the descriptors were matching to multiple tiles at once. Following this, we turned to using canny edge detection to identify the outer edges of the hexagonal game tiles. Though the output of the canny edge detection makes it difficult to identify individual tiles, it will help us in future steps to determine the orientation and vertices of tiles identified by the tile YOLO model. A collab notebook with this work can be found [here](https://colab.research.google.com/drive/1z38mCCmV4SUXURqhSryijEIh4GSuPhHW#scrollTo=JSd6osiKzpQ5&uniqifier=1).

### YOLO models
#### Setup conda environment
```
# From BASE conda environment and SettlersEye directory
conda env create -f environment.yml
# Then activate the created environment for the other steps
```
#### Tile Detection
For identifying tile game pieces, we used a YOLOv8 detection model that we trained using 24 labeled tile images. The model had 7 false positives and 0 false negatives during the testing phase. However, the false positives can be easily removed by setting a high confidence threshold since each tile had a confidence value of at least .95 while each false identification had a confidence value below .90. For data visualizations and example detections, look in TileDetection/runs/detect/train. Within the TileDetection directory, run tile detection with the following command:
```
python detect_tiles.py [-h] --image IMAGE [--image_size IMAGE_SIZE] [--confidence CONFIDENCE]
```
#### Piece detection


For piece detection in our project, we decided to use a YOLOv8 Neural Network for both segmentation and detection. The only pre-processing we did to our data was labeling our images as well as resizing the images to 640x640. Additionally, to remove biases in the training of the model, we added 2 additional copies of each image in the training data, one that is rotated 90 degrees and another rotated 180 degrees. In theory, this should make the model more robust, especially given that we don’t have a huge number of pictures. Altogether, we had a few thousand labels that we fed to the YOLOv8 model for training. 


##### Example piece detection
```
# From SettlersEye directory
cd YOLO
# basic structure: python model.py [-h] --image IMAGE [--image_size IMAGE_SIZE] [--confidence CONFIDENCE]
 python model.py [-h] --image ../dataset/valid/images/IMG_2083-1-_png_jpg.rf.fbb9923145af96e65b3f32ced169181f.jpg
```


#### YOLOv8 Overview
We chose YOLOv8 because of its speed and relatively high accuracy, the testing we have performed pretty well and can probably be improved pretty quickly given a few more images to the dataset. Because we are not really all that worried about the speed of the computation, it is possible that something like Faster R-CNN may be better for accuracy so we are planning on comparing the accuracy of the two models but we initially chose YOLO because we found some good resources on the algorithm and a simple implementation. We also thought the single-shot nature of it was interesting and thought the YOLO accuracy was high enough for our relatively stable task of detecting catan pieces.

As YOLO is a single-shot CNN, there is no way to extract the segmentation from the detection but generally, the algorithm works by taking images and running it through a series of convolution layers followed by a series of fully connected layers. The example they use in the YOLO paper is 24 convolution layers and 2 connected layers as shown in figure 1.

![Screenshot 2023-11-05 184209](https://github.com/lbraby/SettlersEye/assets/97127684/06b7c6df-cf12-4909-b0f4-4d5059646a24)


Figure 1


The output in the classical case is a 30 element vector, 20 for the probability of the prediction being each of the 20 original classifiers in the paper and then two predictions, each with 5 traits, confidence and x, y, width, and length of the detection box. Then to get the prediction you want when training, you take the prediction with the best intersection over union value with the known detection box.

An example of how the model works also comes from the paper, as shown in figure 2.

![Screenshot 2023-11-05 185210](https://github.com/lbraby/SettlersEye/assets/97127684/45393186-6df9-4862-a150-7d365f9ae5d7)


Figure 2


Basically, the model works by splitting the figure into a grid and making a prediction about what each box in the grid contains. It also creates bounding boxes around potential objects. Then, using intersection over union to reduce the number of bounding boxes and the probability of each grid cell containing a class item, it creates a final detection as shown at the end image.


### Contributions of Each Team Member:
- Matthew Carbonaro: YOLO research/overview and game piece detection. Environment creation and first model training attempt.
- Luke Braby: traditional FEX techniques and tile game piece YOLO model

## Part 4:
#### Synthetic Data Generation
To generate synthetic data for training purposes, we first took images of each piece at a few different rotations from a top-down angle. We then created segmentation masks of each image which creates a mask we will use later to mask out the background of the piece images. We additionally took some photos of empty boards which we will place pieces onto. After we had the segmentation masks, we used openCV to mask away the background, make the background transparent, and crop the images so they were the size of the piece, saving them as a png. This cropping makes it simple to use the image size for the bounding boxes in the synthetic images later. Once we had the cropped piece images, we then picked images at random and used the alpha channel to only put the piece on the background that we chose. To improve training reliability, we multiplied the size by a random factor as well as applied a random rotation to the piece before we placed it. We then used the image size as a bounding box, taking into account the rotation and scaling factor, and saved that as a json file that accompanied the synthetic image that was created. Each image has 10 and 60 pieces placed on the board. Once we had those images and the labels, we uploaded it to our roboflow project to combine the images with out real dataset and downloaded the total dataset in the YOLOv8 format. We then trained a new model called `synthetic_model.py` that is stored in the top-level directory.

##### Example of how to generate data
```
# Be in SettlersEye base directory and have activated our conda environment
./synthesis/create_masks.py --dir synthetic_seed/blue -o output -c # Generate and crop masks for each piece type
./synthesis/create_masks.py --dir synthetic_seed/orange -o output -c
./synthesis/create_masks.py --dir synthetic_seed/white -o output -c
./synthesis/create_masks.py --dir synthetic_seed/red -o output -c

 # Generate 50 synthetic images as well as annotations. Should find them in "synthetic_images" directory in SettlersEye
./synthesis/generate_data.py -b synthetic_seed/boards/board1.jpg -d output -n 50
```

#### Board graph construction
To create a usable graph from the hexagonal tiles detected from the YOLO model trained in part 3 (justification for that model choice included in part 3), we had to do some simple geometry to work backwards from bounding boxes to individual hexagons and finally to a graph. As a reminder, the tile detector correctly identified 57 tiles and wrongly identified 7 tiles in the validation set of 3 board images. This means that each tile was identified with some false positives, though these false positives can be easily ignored by setting the identification threshold to 95% confidence (no improvement is needed due to this fact since it performs perfectly on test and validation images when this confidence value is used as a cutoff for detections). No accuracy rate was returned for the test images due to the nature of YOLO as a one look trainer. IoU was not computed since the labeled imagery used polygon labels whereas the output used bounding boxes for detections.  
This paragraph will explain how the graphs used for scoring the game are created. You may also look at TileDetection/createTileGrid.py for the steps below in code form. After detecting tiles using a 95% confidence threshold, make sure that we have 19 tiles detected. Next we locate the neighbors of each tile by finding the tiles that are adjacent to each tile. Then, we use our geometry knowledge to deduce the 6 vertices of each detected tile. By looking at a tile's neighbor, we can set the apothem equal to the distance between the tile and it's neighbor divided by two. The radius of the hexagon can be found by dividing the apothem by cos(30). Using the radius, the angle that the apothem is at, and the centerpoint we can find the 6 vertices with a few simple rotations. Now that we have the vertices, we can merge vertices that are close to eachother into a single vertex so that our graph is connected. With the vertices of our graph known, we can find the edges by looping through all the vertices and pairing vertices whose distance between themselves is equal to the average edge length of our hexagons' sides (with some leway). This graph where the vertices represent tile vertices and the edges are tile edges will be used to validate that settlements and cities are properly placed on the board when they are detected and to make another graph for longest road calculations.  
To make the graph for longest road, vertices are the centerpoins of tile edges and edges are drawn where the distances between these centerpoints is equal to the average edge length of our hexagons' sides (with some leway). This graph is represented by an adjacency list for the sake of the longest road algorithm.

### Contributions of Each Team Member:
- Matthew Carbonaro: Synthetic data generation and model training.
- Luke Braby:  board graph and road graph contruction using detected roads and geommetry
