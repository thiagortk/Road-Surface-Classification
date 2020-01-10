# Road surface quality classification

Here, as the first step, the surface type classification model created in the "Road Surface Classification" folder is used.
You can use the ready-made models used in our experiments and brows in RTK dataset frames with:

```
python testRTK.py PATH_TO_YOUR_FRAMES_SEQUENCE NAME_YOUR_VIDEO_FILE.avi
```

or in KITTI frames with:

```
python testKITTI.py PATH_TO_YOUR_FRAMES_SEQUENCE NAME_YOUR_VIDEO_FILE.avi
```

or in CaRINA frames with:

```
python testCaRINA.py PATH_TO_YOUR_FRAMES_SEQUENCE NAME_YOUR_VIDEO_FILE.avi
```

This is only because of the different resolutions in each dataset, for the placement and size of the result texts (which are fixed and not adaptable to each resolution). The prediction part in the codes is the same.

To run with images from other datasets with different resolutions, you can run on any of the tests sources, and perhaps adapt for results to be displayed at the most appropriate resolution or to automatically adapt.

---------------------------------------------------------------------------------------------

If you want to do a new training and create a new model, populate the folders(class) in the "training_data_asphalt_quality", "training_data_paved_quality" and "training_data_unpaved_quality" folders, and use:

```
python trainAsphaltQuality.py
python trainPavedQuality.py
python trainUnpavedQuality.py 
```
