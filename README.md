# skytemplates_wedge_mask
## Codes for visualize and assess the anomaly. Code for apply an additional bitmask to the BPM

*To mask the CCD3 wedge, shown in skytemplates (time range when CCD2 is dead)*

First, assess the CCD. Look at the data, the slope of the variation, as well
as the extent of it. Through the Jupyter notebook determine the different
sections of the whole mask (as it was recommended to construct it in base of
rectangles).

Once the table for the mask is write out as a txt (space-separated values)
file, add a column with the flag to be used for each one of the sections,
based in the following definitions

Flag | Value | Comment
-----|-------|--------
BPMDEF_FLAT_MIN | 1 | Pixels that are hot in the flats.
BPMDEF_FLAT_MAX | 2 | Pixels that are in the BPM for the flats.
BPMDEF_FLAT_MASK | 4 | Pixels that are hot in the biases.
BPMDEF_BIAS_HOT | 8 | Pixels that are warm in the biases.
BPMDEF_BIAS_WARM | 16 | Pixels that are in the BPM for the biases.
BPMDEF_BIAS_MASK | 32 | Pixels that are downstream of a hot pixel in the bias.
BPMDEF_BIAS_COL | 64 | Pixels on the glowing edges of the CCD.
BPMDEF_EDGE | 128 | Correctable pixels (usually downstream of hot pixels).
BPMDEF_CORR | 256 | Imperfect calibration or excess noise, such as tape bumps.
BPMDEF_SUSPECT | 512 | Ignore for highest-precision work.
BPMDEF_FUNKY_COL | 1024 | Columns with charge redistribution in sky exposures.
BPMDEF_WACKY_PIX | 2048 | Outliers in stacked sky exposures.
BPMDEF_BADAMP | 4096 | Pixel on non-functional amplifier.
BPMDEF_NEAREDGE | 8192 | Suspect due to edge proximity.
BPMDEF_TAPEBUMP | 16384 | Suspect due to known tape bump.

Then, with the table already constructed, use the code `bitmask_alter.py` to
add the mask into the BPM FITS file, looking for not to duplicate the bits.

A typical table containing the definition for the new mask should be like

```
x1 y1 x2 y2 bpmdef
0 1010 90 1810 BPMDEF_SUSPECT
0 1810 120 1970 BPMDEF_SUSPECT
0 1970 560 2900 BPMDEF_SUSPECT
0 2900 770 3240 BPMDEF_NEAREDGE
0 3240 215 3940 BPMDEF_SUSPECT
```

And a typical call of `bitmask_alter.py` would be
`python bitmask_alter.py --bpm {BPM.fits} --tab {table.txt}` 

*It is important to note that the wedge has some dependence with the 
wavelenght. Therefore, one mask per band must be constructed*

### Images for testing the updated mask
As this *wedge* appears when CCD02 was dead and CCD03 alive, we need to test 
the mask. As the mask is constructed from the analysis of the skytemplates, 
the testing should be done on a stacked image, for different bands, on the
same time period.

