# PlotDraw

PlotDraw is naive image vectorisation implementation, for mechanical plot drawers

## Dependencies

OpenCV, Python 3, PIL/Pillow, noise

## Usage

Put images in /image folder

```bash
$ python3 main.py --file file1.png --hatch --darks 
```

Play around with --cannyHigh, --cannylow, --distance, --colors for each image.

```python
usage: main.py [-h] [--preview] [--hatch] [--darks] [-v] [--distance DISTANCE]
               [--cannylow CANNYLOW] [--cannyhigh CANNYHIGH] [--colors COLORS]
               [--hatchdist HATCHDIST] [--file FILE]

Process an image into a vectorised line format.

optional arguments:
  -h, --help            show this help message and exit
  --preview             preview canny edges before
  --hatch               add hatching
  --darks               add darks
  -v, --verbose         verbose logging
  --distance DISTANCE   distance to connect contours by
  --cannylow CANNYLOW   lower threshold for canny
  --cannyhigh CANNYHIGH
                        Upper threshold for canny
  --colors COLORS       num of colors to reduce by
  --hatchdist HATCHDIST
                        Distance to hatch by
  --file FILE           img file name
```

