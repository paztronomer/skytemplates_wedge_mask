''' Code to modify the existing Bad Pixel Masks by adding new masked regions.
Written for Python 2.7 (as this is the supported eups version so far). If want
to migrate to Python 3.+ test the multiprocessing call (the one with partial)
'''

import os
import sys
import time
import logging
import argparse
import numpy as np
from functools import partial
import multiprocessing as mp
try:
    import matplotlib.pyplot as plt
except:
    pass
import fitsio

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# BPM definitions
# From imdetrend::mask_bits.h
bpmdef = dict()
bpmdef.update({'BPMDEF_FLAT_MIN' : 1})
#Pixels that are hot in the flats.
bpmdef.update({'BPMDEF_FLAT_MAX' : 2})
#Pixels that are in the BPM for the flats.
bpmdef.update({'BPMDEF_FLAT_MASK' : 4})
# Pixels that are hot in the biases.
bpmdef.update({'BPMDEF_BIAS_HOT' : 8})
# Pixels that are warm in the biases.
bpmdef.update({'BPMDEF_BIAS_WARM' : 16})
# Pixels that are in the BPM for the biases.
bpmdef.update({'BPMDEF_BIAS_MASK' : 32})
# Pixels that are downstream of a hot pixel in the bias.
bpmdef.update({'BPMDEF_BIAS_COL' : 64})
# Pixels on the glowing edges of the CCD.
bpmdef.update({'BPMDEF_EDGE' : 128})
# Correctable pixels (usually downstream of hot pixels).
bpmdef.update({'BPMDEF_CORR' : 256})
# Imperfect calibration or excess noise, such as tape bumps.
# Ignore for highest-precision work.
bpmdef.update({'BPMDEF_SUSPECT' : 512})
# Columns with charge redistribution in sky exposures.
bpmdef.update({'BPMDEF_FUNKY_COL' : 1024})
# Outliers in stacked sky exposures.
bpmdef.update({'BPMDEF_WACKY_PIX' : 2048})
# Pixel on non-functional amplifier.
bpmdef.update({'BPMDEF_BADAMP' : 4096})
# suspect due to edge proximity.
bpmdef.update({'BPMDEF_NEAREDGE' : 8192})
# suspect due to known tape bump.
bpmdef.update({'BPMDEF_TAPEBUMP' : 16384})
# End of BPMDEF
#
# Change data type to unsigned integers, to match the dtype of the BPMs
#
for k in bpmdef:
    bpmdef[k] = np.uint(bpmdef[k])


def open_fits(fnm, ):
    ''' Open the FITS, read the data and store it on a list
    '''
    tmp = fitsio.FITS(fnm)
    data = []
    for extension in tmp:
        ext_tmp = np.copy(extension.read())
        data.append(ext_tmp)
    if (len(data) == 0):
        logging.error('No extensions were found on {0}'.format(fnm))
        exit(1)
    elif (len(data) >= 1):
        return data

def open_txt(fnm, Nskip):
    ''' Open the text file containing the 2 coordinates per box (bottom-left
    and top-right). Skip the header lines, if any
    The coordinates must be integer, and all rows must have the same number
    of columns
    '''
    # If I don't specify the names key on dtype, then the output will be an
    # array with one more dimension, to harbor the columns. Below array has
    # shape (N,) but if no names are inputs, the shape would be (N, 4)
    m_kw = {
        'names' : ('x1', 'y1', 'x2', 'y2', 'flag'),
        'formats' : ('i4', 'i4', 'i4', 'i4', '|S50'),
    }
    m = np.loadtxt(fnm, dtype=m_kw, comments='#',
                   converters=None, skiprows=Nskip, usecols=None,
                   unpack=False, ndmin=0)
    return m

def get_dict_key(kdict, xval):
    ''' Returns the key associated to the value, for the input dictionary.
    It the key is not in the dictionary, returns None
    '''
    res = [nm for nm, val in kdict.iteritems() if val == xval]
    if (res == []):
        return None
    else:
        return res

def flatten_list(list_2levels):
    ''' Function to flatten a list of lists, generating an output with
    all elements ia a single level. No duplicate drop neither sort are
    performed
    '''
    f = lambda x: [item for sublist in x for item in sublist]
    res = f(list_2levels)
    return res

def bit_count(int_type):
    ''' Function to count the amount of bits composing a number, that is the
    number of base 2 components on which it can be separated, and then
    reconstructed by simply sum them. Idea from Wiki Python.
    Brian Kernighan's way for counting bits. Thismethod was in fact
    discovered by Wegner and then Lehmer in the 60s
    This method counts bit-wise. Each iteration is not simply a step of 1.
    Example: iter1: (2066, 2065), iter2: (2064, 2063), iter3: (2048, 2047)
    Inputs
    - int_type: integer
    Output
    - integer with the number of base-2 numbers needed for the decomposition
    '''
    counter = 0
    while int_type:
        int_type &= int_type - 1
        counter += 1
    return counter

def bit_decompose(int_x):
    ''' Function to decompose a number in base-2 numbers. This is performed by
    two binary operators. Idea from Stackoverflow.
        x << y
    Returns x with the bits shifted to the left by y places (and new bits on
    the right-hand-side are zeros). This is the same as multiplying x by 2**y.
        x & y
    Does a "bitwise and". Each bit of the output is 1 if the corresponding bit
    of x AND of y is 1, otherwise it's 0.
    Inputs
    - int_x: integer
    Returns
    - list of base-2 values from which adding them, the input integer can be
    recovered
    '''
    base2 = []
    i = 1
    while (i <= int_x):
        if (i & int_x):
            base2.append(i)
        i <<= 1
    return base2

def bit_superpose(box_i,
                  bpm=None,
                  aux_bit1=None,
                  aux_bit2_decomp=None,
                  non_masked_val=None,
                  mask_unmasked=None):
    ''' Function to stack bits. Intended to be parallelized inside joint_mask()
    This function works in one box at a time, one flag at a time. The only
    variable that iterates is the box_i_msk
    Returns
    - mask_box: a mask of the same dimensions as the input BPM array. The only
    unmasked values are the one inside the current box
    '''
    logging.info('Working in box: {0}'.format(box_i))
    # Mask all numerical values, then unmask only the pixels on which to work
    # This works: mask_box = np.ma.masked_where(bpm is not np.nan, bpm)
    mask_box = np.ma.array(np.copy(bpm), mask=True)
    if (mask_box.all() is not np.ma.masked):
        logging.warning('The masking of all pixels failed')
    # Unmask the box region to operate on it.
    # NOTE: np.ma.nomask assign zeros instead of recover original data
    mask_box[box_i['y1']:box_i['y2'] ,
             box_i['x1']:box_i['x2']] = mask_box[box_i['y1']:box_i['y2'] ,
                                                 box_i['x1']:box_i['x2']].data
    # Inside the unmasked section, look for each of the bits and its
    # correspondence in the pixels.
    # Add non-masked values to the equation
    for idx, unib in enumerate(list(aux_bit1) + [non_masked_val]):
        # Copy the mask of the box, and operate on the unique bits
        mask_bitx = np.ma.masked_where(mask_box != unib, mask_box, copy=True)
        # Control flow, if there are not non-masked values
        if (np.ma.count(mask_bitx) == 0):
            continue
        if (unib != 0):
            # Get the composing bits for the value, and check if the number
            # we want to assign is already on the components of the pixel
            # bad pixel mask
            unique_bit_comp = aux_bit2_decomp[idx]
            # Control flow, if value is already in the bit decomposition
            # Check for non-masked values also
            if (bpmdef[box_i['flag']] in unique_bit_comp):
                continue
            # Add the flag to the entire unmasked region
            mask_bitx += bpmdef[box_i['flag']]
            # The values from mask_bitx must go to mask_box. Then the
            # values from mask_box must go to the BPM array
            mask_box[~mask_bitx.mask] = mask_bitx[~mask_bitx.mask]
        elif (unib == non_masked_val) and (mask_unmasked):
            # Here we take care of the non-masked pixels
            mask_bitx += bpmdef[box_i['flag']]
            # The values from mask_bitx must go to mask_box. Then the
            # values from mask_box must go to the BPM array
            mask_box[~mask_bitx.mask] = mask_bitx[~mask_bitx.mask]
    return mask_box

def joint_mask(bpm,
               box_msk,
               non_masked_val=0,
               mask_unmasked=True,
               Nproc=None):
    ''' Applies the box-defined mask to the original BPM, avoiding replace
    bits values. The flag need to have an entry in BPMDEF.
    The bit comparison is carried out in parallel, one process per box.
    To be used: ['BPMDEF_SUSPECT', 'BPMDEF_NEAREDGE']
    Inputs
    - bpm: BPM mask array
    - box_msk: numpy structured array corner-coordinates (left-bottom,
    right-top) for the different boxes
    - non_masked_val: value for the non-masked pixels
    - mask_unmasked: whether to mask unmasked pixels. If False then only add
    the flag value to the already masked pixels
    - Nproc: number of processes to be run in parallel for the bit comparison,
    one per box
    '''
    # Check the bits
    #
    # Unique bits in the mask not considering the zeros
    # Transform to tuple for protection
    aux_bit1 = np.unique(bpm[np.where(bpm != non_masked_val)].ravel())
    aux_bit1 = tuple(aux_bit1)
    if (not len(aux_bit1)):
        logging.warning('No masking was detected (values != 0)')
        return False
        # continue
    # Decomposing the above, in base-2 bits
    aux_bit2_decomp = tuple(map(bit_decompose, aux_bit1))
    aux_bit2_count = [len(x) for x in aux_bit2_decomp]
    aux_bit2 = flatten_list(aux_bit2_decomp)
    aux_bit2 = np.unique(aux_bit2)
    # Check if these bits are in the dictionary of BPMDEF
    aux_xm_key = [get_dict_key(bpmdef, y) for y in aux_bit2]
    # If the list contains None, then some of the values weren't found
    if aux_xm_key.count(None):
        logging.warning('Some values in the mask are not defined in BPMDEF')
        aux_xm_key = [x for x in aux_xm_key if x is not None]
    # Flatten the list, because it is composed by lists inside the larger list
    aux_xm_key = flatten_list(aux_xm_key)
    logging.info('Found keys from BPMDEF: {0}'.format(aux_xm_key))
    #
    # Superpose masks, do not duplicate the flag, if already in the bits
    # composing the pixel mask value
    #
    bpm_twin = np.copy(bpm)
    # ====Run in parallel====
    # For Python 3+ the map can be applied in a easier way
    kw_super = {
        'bpm' : bpm,
        'aux_bit1' : aux_bit1,
        'aux_bit2_decomp' : aux_bit2_decomp,
        'non_masked_val' : non_masked_val,
        'mask_unmasked' : mask_unmasked,
    }
    t0 = time.time()
    if (Nproc is None):
        Nproc = len(box_msk)
    P1 = mp.Pool(processes=Nproc)
    # Using the partial() method define an auxiliary function with the fixed
    # values. Then, in the map the iterating value will be provided
    part_bit_superpose = partial(bit_superpose, **kw_super)
    # The map blocks until the result is ready. If want to use apply_async()
    # or other method, processes could be freed when ends, without wait for
    # the others
    mask_i_box = P1.map(part_bit_superpose, box_msk)
    t1 =  time.time()
    txt_time = 'Time in comparing bits: {0:.2f} min'.format((t1 - t0) / 60.)
    logging.info(txt_time)
    # =======================
    # Replace values on the BPM array
    for i_box in mask_i_box:
        bpm_twin[~i_box.mask] = i_box[~i_box.mask]

    #
    # Diagnosis plot
    #
    # Stacked mask
    if True:
        plt.close('all')
        fig = plt.figure(figsize=(5, 8))
        ax0 = fig.add_subplot(1, 1, 1)
        cmap = plt.cm.tab20
        a = np.ma.masked_where(bpm_twin == 0, bpm_twin)
        cmap.set_bad(color='white')
        im0 = ax0.imshow(a, cmap='tab20', origin='lower', vmin=256)
        plt.colorbar(im0)
        plt.show()
    #
    # Plot mask, showing in colormap the number of bits per pixel
    if False:
        aux_bpm = np.copy(bpm)
        for ind, item in enumerate(aux_bit1):
            aux_bpm[np.where(aux_bpm == item)] = aux_bit2_count[ind]
        plt.close('all')
        fig = plt.figure(figsize=(4, 8))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(aux_bpm, origin='lower', cmap='viridis_r')
        plt.colorbar(im)
        plt.show()
        del aux_bpm


if __name__ == '__main__':
    desc_text = 'Per-CCD code. Based on BPM, and box coodinates'
    desc_text += ' (x1, y1, x2, y2) for new masks to be included on the'
    desc_text += ' initial BPM'
    abc = argparse.ArgumentParser(description=desc_text)
    h0 = 'Image to be loaded, to check the masking'
    tmp_pca = 'Y4A1_20160801t1215_g_c03_r2930p01_skypca-tmpl.fits'
    abc.add_argument('--image', help=h0, default=tmp_pca)
    h1 = 'BPM FITS file'
    tmp_bpm = 'D_n20170815t0824_c03_r3370p01_bpm.fits'
    abc.add_argument('--bpm', help=h1, default=tmp_bpm)
    h2 = 'Space-sepatared file with 2 vertices (bottom-left and top-right) per'
    h2 += ' box, to be be included in the BPM masking. Format should be: x_ini'
    h2 += ' y_ini x_end y_end bpmdef_flag'
    tmp_add = 'vert_flag_op3.txt'
    abc.add_argument('--tab', help=h2, default=tmp_add)
    h3 = 'How many lines skip at the top of the space-separated file defining'
    h3 += ' the boxes. Default: 1'
    abc.add_argument('--skip', help=h3, default=1, type=int)
    h4 = 'How many processes to run in parallel for the mask stacking. Default'
    h4 += ' is the number of boxes'
    abc.add_argument('--n1', help=h4, type=int)
    # Variables
    abc = abc.parse_args()

    aux_pca = open_fits(abc.image)
    aux_bpm = open_fits(abc.bpm)
    aux_tab = open_txt(abc.tab, abc.skip)
    # We expect one extension.
    if (len(aux_bpm) != 1):
        logging.error('This code is implemented to update one-extension only')
        exit(1)
    # Call different flags, for different boxes. One flag at a time
    aux_mask = joint_mask(aux_bpm[0], aux_tab, Nproc=abc.n1)

    # print(bpmdef.keys(), bpmdef.values())
