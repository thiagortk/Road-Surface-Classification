# Road surface classification only (asphalt, paved or unpaved)

Here is only for Road Surface Classification (asphalt, paved or unpaved). If you want to train a new model for Road Surface Quality Classification, you need to train the surface type classification model first, available here.

You can use the ready-made model used in our experiments and brows in your frames with:

```
python test.py PATH_TO_YOUR_FRAMES_SEQUENCE NAME_YOUR_VIDEO_FILE.avi
```

If you want to do a new training and create a new model, populate the folders(class) in the "training_data" folders, and use:

```
python train.py
```
