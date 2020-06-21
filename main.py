
import argparse
import cv2
import math  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from PIL import ImageDraw
import noise

def displayImages(files):
    f = []
    for i, im in enumerate(files):
        fig = plt.figure(i)
        plt.imshow(im,  cmap='Greys_r')
        f.append(fig)
    plt.show(block=False)
    plt.waitforbuttonpress(0) # this will wait for indefinite times
    plt.close('all')
def increaseContrast(img):
    constant = 0.5
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            img[x][y] = img[x][y] - (img[x][y] * constant)
            if (img[x][y] > 255): img[x][y] = 255
            if (img[x][y] < 0): img[x][y] = 0
    return img

def getEdges(img, cannyLower = 50,  cannyHigher = 100, preview = False):
    print("Getting Edges...")
    shape = img.shape
    height, width = shape
    im = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(im, cannyLower, cannyHigher)
    if preview:
        plt.imshow(edges, cmap='Greys_r')
        plt.show()
    return edges

def getDots(img):
    print("Getting Dots...")
    shape = img.shape
    height, width = shape
    rows = []
    for x in range(height):
        for y in range(width):
            if img[x,y] == 255:
                rows.append((x,y))
    return rows

def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return math.sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2))

def getLines(dots):
    print("Finding Contours...")
    straightLines = []
    line = []
    for dot in dots:
        x, y = dot
        if (len(line) == 0):
            line.append((x,y))
        elif (line[-1][1] == y - 1):
            line.append((x,y))
        else:
            if (len(line) != 1):
                straightLines.append([(line[0][0],line[0][1]), (line[-1][0],line[-1][1])])
            line = []
    return straightLines

def getContours(lines):
    print("getting contours")
    contours = []
    usedlines = []
    templines = lines.copy()
    while len(templines) > 0:
        contour = []
        line = templines.pop(0)
        contour.append(line[0])
        contour.append(line[-1])
        i = 0
        while i < len(templines):
            line2 = templines[i]
            xdist = abs(line2[0][0] - line[-1][0])
            ydist =  abs(line2[0][1] - line[-1][1])
            same = line[0][0] == line2[0][0] & line[-1][0] == line2[-1][0]
            if ((ydist <= 1) & (xdist <= 1) & (same == False)):
                contour.append(line2[0])
                line.append(line2[-1])
                contour.append(line2[-1])
                templines.pop(i)
            else:
                i += 1
        contours.append(contour)
    return contours

def combineContours(lines, threshold):
    print("Combining lines...")
    contours = []
    usedlines = []
    templines = lines.copy()
    while len(templines) > 0:
        contour = []
        line = templines.pop(0)
        contour.append(line[0])
        contour.append(line[-1])
        i = 0
        while i < len(templines):
            line2 = templines[i]
            distance = dist(line2[0], line[-1])
            same = line[0][0] == line2[0][0] & line[-1][0] == line2[-1][0]
            if ((distance <= threshold) & (same == False)):
                avgX = (contour[-1][0] + line2[0][0])/ 2
                avgY = (contour[-1][1] + line2[0][1])/ 2
                
                avgA = ((contour[-1][0] + avgX)/ 2, (contour[-1][1] + avgY)/ 2)
                avgB = ((line2[0][0] + avgX)/ 2, (line2[0][1] + avgY)/ 2)
                # Uncomment this and comment the line above for some interesting results? ?
                #  avgB = ((contour[0][0] + avgX)/ 2, (contour[0][1] + avgY)/ 2)

                contour[-1] = (avgX, avgY)
                contour.append((avgX, avgY))
                
                contour.append(line2[-1])
                line.append(line2[-1])
                templines.pop(i)
            else:
                i += 1
        contours.append(contour)
    return contours

def plotLines(img, lines):
    print("Plotting Lines...")

    for line in lines:
        
        plt.plot([line[0][0], line[-1][0]], [line[0][1],line[-1][1]])
    plt.show()

def plotContours(img, contours, verbose = False):
    print("Plotting contours...")
    draw = ImageDraw.Draw(img)

    for n, contour in enumerate(contours):
        length = len(contour)
        i = 0
        while (i < length - 1):
            # plt.plot(
            #     [contour[i][0], contour[i+1][0]], [contour[i][1],contour[i+1][1]],
            #     color='black', linewidth = 1
            # )
            draw.line(((contour[i][0],contour[i][1]),(contour[i+1][0],contour[i+1][1])), fill=(0,0,0), width=math.floor(genPerlinNoise(contour[i][0], contour[i][1])/5) + 2)
            i += 1 
        if verbose: print(n , len(contours), '- ', length)

def plotHatches(img, contours, verbose = False):
    print("Plotting contours...")
    draw = ImageDraw.Draw(img)

    for n, contour in enumerate(contours):
        length = len(contour)
        i = 0
        while (i < length - 1):
            # plt.plot(
            #     [contour[i][0], contour[i+1][0]], [contour[i][1],contour[i+1][1]],
            #     color='black', linewidth = 0.8
            # )
            draw.line(((contour[i][0],contour[i][1]),(contour[i+1][0],contour[i+1][1])), fill=(0,0,0), width=1)
            i += 1
        if verbose: print(n , len(contours), '- ', length)
# rotate +90deg
def rotateLine90(line, shape):
    height, width = shape
    return [(line[0][1], 0 - line[0][0] + height), (line[-1][1], 0 - line[-1][0] + height)]

def rotateContours90(contours, shape):
    print("Rotating Contours...")
    height, width = shape
    rotContours = []
    for contour in contours:
        rotContour = [];
        length = len(contour)
        i = 0
        while (i < length):
            rotContour.append(rotateLine90([contour[i], contour[i+1]], shape))
            i += 2
        flattened_list = [y for x in rotContour for y in x]

        rotContours.append(flattened_list)
    return rotContours;

# rotate +45deg
def rotateLine45(line, shape):
    height, width = shape

    x = line[-1][0]
    y = line[-1][1] 

    newx1 = (x * math.cos(270 * math.pi / 180)) - (y * math.cos(270 * math.pi/180)) 
    newy1 = (x * math.cos(270 * math.pi / 180)) + (y * math.cos(270 * math.pi/180))

    x = line[0][0]
    y = line[0][1] 
    newx2 = (x * math.cos(270 * math.pi / 180)) - (y * math.cos(270 * math.pi/180)) 
    newy2 = (x * math.cos(270 * math.pi / 180)) + (y * math.cos(270 * math.pi/180))

    return [(newx2, newy2), (newx1, newy1)]

def rotateContours45(contours, shape):
    print("Rotating Contours...")
    height, width = shape
    rotContours = []
    for contour in contours:
        rotContour = [];
        length = len(contour)
        i = 0
        while (i < length):
            rotContour.append(rotateLine45([contour[i], contour[i+1]], shape))
            i += 2
        flattened_list = [y for x in rotContour for y in x]

        rotContours.append(flattened_list)
    return rotContours;

def hatch(img, disthatch):
    print('hatching...')
    h, w = img.shape
    lines = []
    x = 0
    while x < h:
        currentLine = []
        y = 0
        while y < w:
            if img[x][y] <= 85:
                if len(currentLine) == 0:
                    currentLine.append((x + genPerlinNoise(x, y) ,y + genPerlinNoise(x, y)))
                elif len(currentLine) >= 1:
                    currentLine.append((x + genPerlinNoise(x, y) ,y + genPerlinNoise(x, y)))
                    currentLine.append((x + genPerlinNoise(x, y) ,y + genPerlinNoise(x, y)))
            else:
                # off by 1? eh
                if len(currentLine) >= 1:
                    currentLine.append((x + genPerlinNoise(x, y) ,y + genPerlinNoise(x, y)))
                    lines.append(currentLine)
                    currentLine = []
            y += 10
        if len(currentLine) == 1:
            currentLine.append((x,y))
            lines.append(currentLine)
        x += disthatch+ math.floor(genPerlinNoise(x, y))
    
    x = 0
    while x < h + w:
        y = 0
        currentLine = []
        while (y < x) & (y < h):
            if (y + w) <= x: 
                y+= 10
                continue

            if img[y][x - y] <= 125:
                if len(currentLine) == 0:
                    currentLine.append((y + genPerlinNoise(y,x -y) ,x - y + genPerlinNoise(y,x -y)))
                elif len(currentLine) >= 1:
                    currentLine.append((y + genPerlinNoise(y,x -y) ,x - y + genPerlinNoise(y,x -y)))
                    currentLine.append((y + genPerlinNoise(y,x -y) ,x - y + genPerlinNoise(y,x -y)))
            else:
                # off by 1? eh
                if len(currentLine) >= 1:
                    currentLine.append((y + genPerlinNoise(y,x -y) ,x -y + genPerlinNoise(y,x -y)))
                    lines.append(currentLine)
                    currentLine = []
            y += 10
        if len(currentLine) == 1:
            currentLine.append((y,x -y))
            lines.append(currentLine)
        x += disthatch + math.floor(genPerlinNoise(x, y))
    return lines

def getDarks(img):
    print('getting dark areas')
    h, w = img.shape
    lines = []
    x = 0
    while x < h:
        currentLine = []
        y = 0
        while y < w:
            if img[x][y] == 0:
                if len(currentLine) == 0:
                    currentLine.append((x ,y))
                elif len(currentLine) >= 1:
                    currentLine.append((x ,y))
                    currentLine.append((x ,y))
            else:
                # off by 1? eh
                if len(currentLine) >= 1:
                    currentLine.append((x ,y))
                    lines.append(currentLine)
                    currentLine = []
            y += 10
        if len(currentLine) == 1:
            currentLine.append((x,y))
            lines.append(currentLine)
        x += 1
    return lines

def sketch(img, args):

    images = []
    img2 = np.flip(np.rot90(img))

    distance = int(args.distance) if args.distance else 10
    colors = int(args.colors) if args.colors else 4
    cannyLower = int(args.cannylow) if args.cannylow else 50
    cannyHigher = int(args.cannyhigh) if args.cannyhigh else 100
    distHatch = int(args.hatchdist) if args.hatchdist else 4
    
    edges = getEdges(img,cannyLower = cannyLower, cannyHigher = cannyHigher)
    dots = getDots(edges)
    lines = getLines(dots)
    testContour = rotateContours90(lines, img.shape)
    
    edges2 = getEdges(img2, cannyLower = cannyLower, cannyHigher = cannyHigher)
    dots2 = getDots(edges2)
    lines2 = getLines(dots2)
    contours2 = getContours(lines2)
    combinedContours = combineContours( lines2 + testContour, distance)
    img = getReducedColourImage(img, colors)
    im = Image.new('RGBA', img.shape, (255,255,255,255))
    draw = ImageDraw.Draw(im)
    if args.hatch:
        hatchlines = rotateContours90(hatch(img,  distHatch), img.shape)
        if args.verbose: print(len(hatchlines))
        plotHatches(im, hatchlines , verbose = args.verbose)
    if args.darks:
        darklines = rotateContours90(getDarks(img), img.shape)
        if args.verbose: print(len(darklines))
        plotHatches(im, darklines ,  verbose = args.verbose)
    if args.verbose: print(len( combinedContours))

    plotContours(im, combinedContours, verbose = args.verbose)
    plt.imshow(np.asarray(im), origin='lower')
    print('Done!')
    plt.show()

def getReducedColourImage(img, N): 
    h, w = img.shape
    for x in range(h):
        for y in range(w):
            img[x][y] = math.floor((img[x][y] * N/255)) * (255 / N)
    return img
def genPerlinNoise(x, y ):
    scale = 100.0
    octaves = 7
    persistence = 0.5
    lacunarity = 2.0
    perl = noise.pnoise2(x/scale, y/scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=0)
    return ((perl + 1) / 2) * 20

if __name__ == "__main__":
    folderPath = "images/"
    fileName = "file2.png"
    parser = argparse.ArgumentParser(description='Process an image into a vectorised line format.')
    parser.add_argument("--preview", help="preview canny edges before",
                    action="store_true")
    parser.add_argument("--hatch", help="add hatching",
                    action="store_true")
    parser.add_argument("--darks", help="add darks",
                    action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose logging",
                    action="store_true")
    parser.add_argument("--distance", help="distance to connect contours by")
    parser.add_argument("--cannylow", help="lower threshold for canny",)
    parser.add_argument("--cannyhigh", help="Upper threshold for canny",)
    parser.add_argument("--colors", help="num of colors to reduce by",)
    parser.add_argument("--hatchdist", help="Distance to hatch by",)
    parser.add_argument("--file", help="img file name",)

    args = parser.parse_args()
    fileName = args.file if args.file else fileName
    img = cv2.imread("%s%s" % (folderPath, fileName),  cv2.IMREAD_GRAYSCALE)

    if args.preview:
        edges = getEdges(img, preview= True,
        cannyLower = int(args.cannylow), cannyHigher = int(args.cannyhigh))
    else:
        sketch(img, args)

