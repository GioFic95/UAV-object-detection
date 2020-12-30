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

[On YouTube](https://youtu.be/D5o2BuDhtRc) you can find a brief "tutorial"
where I explain how to use this GUI.

---
### Installation and usage:
(you can also use [Anaconda](https://www.anaconda.com/products/individual#Downloads)
instead of pip and virtualenv)

1. install Python from [here](https://www.python.org/downloads/)
   (tested on [Python 3.6](https://www.python.org/downloads/release/python-3612/),
   [install Python 3.6](https://www.pytorials.com/python-download-install-windows/),
   [install Python 3.8](https://www.python.org/downloads/release/python-387/))
   ![Python installer](https://www.pytorials.com/wp-content/uploads/2017/12/python3.6_installation_2.png)\
   **in the installer, make sure "Add Python 3.x to PATH" option is checked;**
2. open your prompt/cmd (on Windows) or terminal/bash (on macOS or Linux) and go to
   your working directory (let's call it `w_dir`), where there are the script `gui.py`,
   the file `requirements.txt` and the folder with the input images (let's call it `in_imgs`):
   ```
   cd w_dir
   ```
2. install pip and virtualenv:
   ```
   python3 -m pip install --user --upgrade pip   # or
   python -m pip install --user --upgrade pip
   
   python3 -m pip install --user virtualenv      # or
   python -m pip install --user virtualenv
   ```
3. create a virtual environment called `labeling`:
   ```
   python3 -m venv labeling   # or
   python -m venv labeling
   ```
   the content of your working directory will something be:
   ```
   - in_imgs
   - labeling
   - gui.py
   - requirements.txt
   ```
4. activate the virtual environment:
   ```
   .\labeling\Scripts\activate    # on Windows or
   source labeling/bin/activate   # on macOS/Linux
   ```
5. install the requirements:
   ```
   pip install -r requirements.txt
   ```
   on Windows, it may be necessary to install the Microsoft C++ Build Tools from
   [here](https://visualstudio.microsoft.com/it/visual-cpp-build-tools/), to make
   the requirements installation work, so pay attention to what is written in the
   prompt/cmd;
7. execute the script:
   ```
   python gui.py -dir ./in_imgs -save results/results.tsv
   ```
   recall that `in_imgs` must be the name of the directory which
   contains the input imagges;
8. submit all the shapes in the images you work on;
9. leave the virtual environment:
   ```
   deactivate
   ```

---
### Notes

Once you installed everything, if you want to resume your work, it is
sufficient to repeat the steps 4, 7, 8, 9.

If you want to continue your labeling work in a different moment, the
script will resume directly from the next picture with respect to the
last one from which you submitted some labels, so, remember to **fill
the labels for all the shapes in a photo** before quitting.

**Blurry images** still need labeling, until the shape and color are
recognizable. **Partial shapes** (targents on the boudary of the image) can
be ignored. For **empty images** (without any target) use the button `empty`.

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
