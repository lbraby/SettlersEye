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
For identifying tile game pieces, we used a YOLOv8 detection model that we trained using 24 labeled tile images. The model had 7 false positives and 0 false negatives during the testing phase. However, the false positives can be easily removed by setting a high confidence threshold since each tile had a confidence value of at least .95 while each false identification had a confidence value below .90. For data visualizations and example detections, look in TileDetection/runs/detect/train. Within the TileDetection directory, run tile detection with the following command:
```
python detect_tiles.py [-h] --image IMAGE [--image_size IMAGE_SIZE] [--confidence CONFIDENCE]
```

### Contributions of Each Team Member:
- Matthew Carbonaro: 
- Luke Braby: traditional FEX techniques and tile game piece YOLO model