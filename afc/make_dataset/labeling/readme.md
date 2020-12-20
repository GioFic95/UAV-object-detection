# Labeling GUI

Based on a work by the Sapienza Flight Team in 2019.

---
### Description

Our dataset is composed of photos taken from a terrace about 30
meters high, with **geometric shapes** made on cardboard on the floor.

Our goal is to provide *labels* for each image in our dataset,
to feed a script which will add **synthetic alphanumeric characters**
on the shapes, to produce images which in turn will feed a Machine
Learning Model.

The labels consist of: shape name, shape color, bounding box, rotation.

The script provides a GUI for easily inserting al kinds of labels.
You can *zoom* to select a portion of the image and better see the
shapes, crop and rotate the shape to find bounding box and rotation,
and select shape name and color from a drop-down list.

[On YouTube](https://youtu.be/JYQC4qPPBAI) you can find a brief "tutorial"
where I explain how to use this GUI.

---
### Installation and usage:
(you can also use [Anaconda](https://www.anaconda.com/products/individual#Downloads)
instead of pip and virtualenv)

1. install Python from [here](https://www.python.org/downloads/)
  (tested on [Python 3.6](https://www.python.org/downloads/release/python-3612/));
2. install pip and virtualenv:
   ```
   python3 -m pip install --user --upgrade pip
   python3 -m pip install --user virtualenv
   ```
3. create and activate a virtual environment:
   ```
   python3 -m venv labeling
   .\labeling\Scripts\activate    # on Windows or
   source labeling/bin/activate   # on macOS/Linux
   ```
4. install the requirements:
   ```
   pip install -r requirements.txt
   ```
5. copy the script `gui.py` and the folder with the input images
   (`in_imgs`) into the `labeling` directory, which must contain:
   ```
   - environment stuff
   - in_imgs
   - gui.py
   ```
6. execute the script:
   ```
   cd labeling
   python gui.py -dir ./in_imgs -save results/results.tsv
   ```
7. submit all the shapes in the images you work on;
8. leave the virtual environment:
   ```
   deactivate
   ```

---
### Notes

If you want to continue your labeling work in a different moment, the
script will resume directly from the next picture with respect to the
last one from which you submitted some labels, so, remember to **fill
the labels for all the shapes in a photo** before quitting. 

You can use the `esc` key to quit (undo) from the zoom/regions dialogue
and `enter` to confirm the selection.

You can use the `tab` key to move across fields, to fill them faster,
and `up/down arrows` to increment/decrement the rotation.

---
### Shortcuts:
- ctrl+M: select shape regions
- ctrl+Left: previous image
- ctrl+Right: next image
- ctrl+S: submit
- ctrl+R: rotate 90 degrees
- ctrl+W: close window
- esc: close zoom/regions dialogue
