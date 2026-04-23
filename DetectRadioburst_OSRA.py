import sys
import os
sys.path.insert(1, './work1/okoro/type3detectosra/')  # use the repo type3detectosra

import matplotlib.dates as mdates
import datetime
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
import scipy
from scipy import interpolate, optimize
from astropy.time import Time
import pandas as pd
import matplotlib as mpl


# Set the matplotlib date epoch to Unix time origin for consistent date handling
mpl.rcParams['date.epoch'] = '1970-01-01T00:00:00'
try:
    mdates.set_epoch('1970-01-01T00:00:00')
except:
    pass

from skimage.transform import probabilistic_hough_line
import Radiotools as rt        


def read_osra(fname):
    """
    Read a full 4-antenna OSRA .roh binary file.

    fname : full path to the .roh file
            e.g. "/net/lyot/scratch3/vocks/OSRA/1998/CD_232/980820_232.roh"

    Each binary record is 1040 bytes:
      bytes 0–5   : BCD-encoded timestamp (year, month, day, hour, min, sec)
      byte  6     : sub-second fraction (units of 100 ms)
      bytes 16–   : spectral data for all four antennas (1024 channels total)

    Returns
    -------
    dyspec : 2D array (n_records, 1024)  — raw spectrogram, all antennas
    t_fits : 1D array of datetime objects
    f_fits : 1D array of frequencies (MHz) for all 1024 channels
    """

    # ── Define frequency arrays for all four OSRA antennas ───────────────────
    # Each antenna covers a different frequency band.
    # The formula reconstructs the linear frequency axis from channel index.
    f1 = 800.0 - np.array(range(256)) * 400. / 256.   # 800  → 400  MHz
    f2 = 400.0 - np.array(range(256)) * 200. / 255.   # 400  → 200  MHz
    f3 = 170.0 - np.array(range(256)) * 70.  / 256.   # 170  → 100  MHz
    f4 = 100.0 - np.array(range(256)) * 60.  / 255.   # 100  →  40  MHz
    f_fits = np.concatenate((f1, f2, f3, f4))          # 1024 channels total

    # ── Determine number of records from file size ────────────────────────────
    # Each record is exactly 1040 bytes.
    # +0.5 before int() rounds to nearest integer instead of truncating.
    file_stats = os.stat(fname)
    print(file_stats)                          # full stat output
    print(file_stats.st_size)                  # file size in bytes
    print(file_stats.st_size / 1040)           # fractional record count (diagnostic)
    a1 = int(file_stats.st_size / 1040 + 0.5) # integer record count

    # ── Allocate output arrays ────────────────────────────────────────────────
    t_fits = np.zeros(a1, dtype=object)          # will hold datetime objects
    dyspec = np.zeros((a1, 1024), dtype=np.ubyte)

    # ── Read records one by one ───────────────────────────────────────────────
    file = open(fname, "rb")
    for i in range(a1):
        data_chunk    = file.read(1040)
        np_data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)

        # BCD decode: each byte stores two decimal digits in its two nibbles.
        # High nibble (upper 4 bits) = tens digit: byte >> 4  = int(byte / 16)
        # Low  nibble (lower 4 bits) = units digit: byte & 0x0F = byte & 15
        year        = int(np_data_chunk[0] / 16) * 10 + (np_data_chunk[0] & 15)
        year        = year + 1900 if year > 50 else year + 2000
        month       = int(np_data_chunk[1] / 16) * 10 + (np_data_chunk[1] & 15)
        day         = int(np_data_chunk[2] / 16) * 10 + (np_data_chunk[2] & 15)
        hour        = int(np_data_chunk[3] / 16) * 10 + (np_data_chunk[3] & 15)
        minute      = int(np_data_chunk[4] / 16) * 10 + (np_data_chunk[4] & 15)
        second      = int(np_data_chunk[5] / 16) * 10 + (np_data_chunk[5] & 15)
        microsecond = 100000 * np_data_chunk[6]   # byte 6 is 100 ms units → microseconds

        print(datetime.datetime(year, month, day, hour, minute, second, microsecond))

        t_fits[i]  = datetime.datetime(year, month, day, hour, minute, second, microsecond)
        dyspec[i, :] = np_data_chunk[16:]   # spectral data starts at byte 16

    
    # After the loop, we have read all records. Now we can close the file.
    file.close()        # ← correct Python syntax

    # ALTERNATIVE: use a context manager to guarantee the file is closed
    # even if an exception occurs mid-read:
    #
    # with open(fname, "rb") as file:
    #     for i in range(a1):
    #         data_chunk = file.read(1040)
    #         ... (same decoding logic)

    return (dyspec, t_fits, f_fits)


def read_osraf2(fname):
    """
    Read a single-antenna (f2 band only) OSRA .roh binary file.

    Compared to read_osra, this function:
      - Uses only antenna f2 (400 → 200 MHz, 256 channels)
      - Slices bytes 272–528 from each record (f2 data offset in the binary layout)
      - Converts t_fits to Python datetime via to_python_datetime() at the end

    fname : full path to the .roh file
            e.g. "/net/lyot/scratch3/vocks/OSRA/2003/CD_300/031027_300.roh"

    Returns
    -------
    dyspec : 2D array (n_records, 256)  — raw spectrogram, f2 band only
    t_fits : 1D array of Python datetime objects
    f_fits : 1D array of 256 frequencies (MHz) for the f2 band
    """

    # ── f2 frequency axis: 400 MHz → 200 MHz over 256 channels ───────────────
    f2     = 400.0 - np.array(range(256)) * 200. / 255.
    f_fits = f2

    # ── Determine number of records from file size ────────────────────────────
    file_stats = os.stat(fname)
    a1 = int(file_stats.st_size / 1040 + 0.5)

    # ── Initialise t_fits with a dummy placeholder ────────────────────────────
    # The array is filled with real timestamps in the loop below.
    # The dummy value is overwritten for every record — it is only here to
    # pre-allocate the array with the correct length before the loop starts.
    dummy_start = np.datetime64('2025-02-01T15:23:15.00')
    t_fits = dummy_start + np.linspace(0, 1, a1).astype('timedelta64[D]')

    # ALTERNATIVE: allocate as an object array of None and fill with datetimes:
    # t_fits = np.empty(a1, dtype=object)

    # ── Allocate spectrogram array for f2 only (256 channels) ─────────────────
    dyspec = np.zeros((a1, 256), dtype=np.ubyte)

    # ── Read records one by one ───────────────────────────────────────────────
    file = open(fname, "rb")
    for i in range(a1):
        data_chunk    = file.read(1040)
        np_data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)

        # BCD timestamp decode — same method as read_osra
        year        = int(np_data_chunk[0] / 16) * 10 + (np_data_chunk[0] & 15)
        year        = year + 1900 if year > 50 else year + 2000
        month       = int(np_data_chunk[1] / 16) * 10 + (np_data_chunk[1] & 15)
        day         = int(np_data_chunk[2] / 16) * 10 + (np_data_chunk[2] & 15)
        hour        = int(np_data_chunk[3] / 16) * 10 + (np_data_chunk[3] & 15)
        minute      = int(np_data_chunk[4] / 16) * 10 + (np_data_chunk[4] & 15)
        second      = int(np_data_chunk[5] / 16) * 10 + (np_data_chunk[5] & 15)
        microsecond = 100000 * np_data_chunk[6]

        t_fits[i]    = datetime.datetime(year, month, day, hour, minute, second,                   microsecond)

        # f2 spectral data occupies bytes 272–527 in each 1040-byte record.
        # bytes 0–15   : header (timestamp)
        # bytes 16–271 : f1 antenna data (256 bytes)
        # bytes 272–527: f2 antenna data (256 bytes)  ← this is what we keep
        # bytes 528–783: f3 antenna data
        # bytes 784–1039: f4 antenna data
        dyspec[i, :] = np_data_chunk[272:528]

    file.close()

    # ALTERNATIVE context manager form:
    # with open(fname, "rb") as file:
    #     for i in range(a1):
    #         data_chunk = file.read(1040)
    #         ... (same decoding)

    # Convert the numpy datetime / object array to plain Python datetime objects
    # for compatibility with matplotlib and scipy interpolation downstream.
    t_fits = to_python_datetime(t_fits)

    return (dyspec, t_fits, f_fits)


def to_python_datetime(arr):
    """
    Convert an array of timestamps to plain Python datetime objects.

    Handles two input types:
      - Already Python datetime objects : returned unchanged.
      - numpy datetime64 or similar     : converted via datetime64[ms].

    This conversion is needed because scipy interpolate.interp1d and
    matplotlib both require either numeric values or plain Python datetimes —
    numpy datetime64 objects cause type errors in those contexts.

    Parameters
    ----------
    arr : array-like
        Array of timestamps in any datetime-compatible format.

    Returns
    -------
    numpy array of Python datetime.datetime objects.
    """
    arr = np.array(arr)

    # If the first element is already a Python datetime, the whole array is —
    # return it as-is without any conversion.
    if isinstance(arr[0], datetime.datetime):
        return arr

    # Otherwise convert: numpy datetime64 → millisecond precision → Python datetime
    return np.array(arr).astype('datetime64[ms]').astype(datetime.datetime)

def idx_val_pos(f_fits, target):
    """
    Find the index in f_fits whose value is closest to target.

    Parameters
    ----------
    f_fits : array-like
        Frequency axis array (MHz).
    target : float
        The frequency value you want to locate.

    Returns
    -------
    int : index of the closest frequency channel.
    """
    # Compute the absolute difference between every channel and the target,
    # then return the index of the minimum difference.
    return np.abs(np.array(f_fits) - target).argmin()


def cut_low(dyspec, f_fits, f_low_cut_val=21):
    """
    Remove all frequency channels below a given threshold.

    Useful for cutting RFI-dominated low-frequency channels from the
    bottom of the spectrogram before detection.

    Parameters
    ----------
    dyspec : 2D array (n_times, n_freqs)
        Raw or preprocessed spectrogram.
    f_fits : 1D array
        Frequency axis (MHz), same length as dyspec axis 1.
    f_low_cut_val : float
        Frequency in MHz below which channels are removed.
        Default is 21 MHz.

    Returns
    -------
    dyspec : 2D array with low-frequency channels removed.
    f_fits : 1D array with corresponding frequencies removed.
    """
    # Find the channel index closest to the cut frequency
    cut_index = idx_val_pos(f_fits, f_low_cut_val)

    # Keep only channels at or above that index
    dyspec = dyspec[:, cut_index:]
    f_fits = f_fits[cut_index:]

    return (dyspec, f_fits)


def preproc2(dyspec, gauss_sigma=(1.5, 0), background_normalize=True):
    """
    Remove slowly varying background and smooth the dynamic spectrum.

    Step 1 (optional): divide each frequency channel by a background
    reference level computed from the quietest 10–30% of time samples,
    then subtract 1 so that background → 0 and burst excess → positive.

    Step 2: apply a Gaussian smoothing filter along the time axis only
    (sigma along frequency = 0 by default), reducing time-axis noise
    while preserving frequency resolution.

    Parameters
    ----------
    dyspec : 2D array (n_times, n_freqs)
        Raw spectrogram — rows are time steps, columns are frequency channels.
    gauss_sigma : tuple (sigma_time, sigma_freq)
        Smoothing widths. Default (1.5, 0) smooths time only.
    background_normalize : bool
        Set False to skip background removal and pass dyspec straight
        through to the smoothing step.

    Returns
    -------
    data_background_removed : 2D array
        Spectrogram after background division, before smoothing.
    data_smoothed : 2D array
        Spectrogram after background division AND Gaussian smoothing.
    """

    if background_normalize:

        # ── Step 1a: sort all time rows at each channel, pick quiet rows ──────
        # Rows 10%–30% of the sorted stack are the quietest time samples.
        # Taking their mean gives one background value per frequency channel.
        n_times    = dyspec.shape[0]
        row_low    = int(n_times * 0.1)
        row_high   = int(n_times * 0.3)

        sorted_in_time           = np.sort(dyspec, axis=0)
        quiet_rows               = sorted_in_time[row_low:row_high, :]
        background_per_channel   = np.nanmean(quiet_rows, axis=0)

        # ── Step 1b: protect against zero or NaN background ───────────────────
        # A dead channel has background = 0; dividing by it gives inf.
        # Replace those values with NaN so the division yields NaN instead,
        # which is then replaced by 0 in the next step.
        background_per_channel = np.where(
            (background_per_channel == 0) | np.isnan(background_per_channel),
            np.nan,
            background_per_channel
        )

        # ── Step 1c: normalise and remove background ──────────────────────────
        # (signal / background) - 1  →  quiet times ≈ 0, bursts > 0
        data_background_removed = (dyspec / background_per_channel) - 1

        # Replace NaN from dead channels with 0
        data_background_removed = np.nan_to_num(data_background_removed, nan=0.0)

    else:
        # No background removal — pass the raw array straight through
        data_background_removed = dyspec

    # ── Step 2: Gaussian smoothing along the time axis ────────────────────────
    # sigma is a tuple matching the array axes (time, frequency).
    # order=0     : plain Gaussian (not a derivative)
    # mode=nearest: edge pixels are extended by repeating the border value
    # truncate=5.0: kernel is cut off beyond 5 sigma (avoids far-edge artefacts)
    data_smoothed = scipy.ndimage.gaussian_filter(
        data_background_removed,
        sigma=gauss_sigma,
        order=0,
        output=None,
        cval=0.0,
        truncate=5.0,
        mode='nearest'
    )

    return (data_background_removed, data_smoothed)


def binarization(data_smoothed, N_order=8, peak_r=0.9993):
    """
    Mark local peaks along the time axis of a smoothed spectrogram.

    A pixel at time index t is kept (set to 1) only if it is strictly
    greater than every one of its N_order-1 neighbours on both sides
    in time, within a tolerance factor peak_r.

    Pixels that fail any comparison are zeroed out. The result is a
    sparse binary map — 1 only at sharp local maxima — which is what
    the probabilistic Hough line transform operates on.

    Parameters
    ----------
    data_smoothed : 2D array (n_times, n_freqs)
        Output of preproc2 — background-removed and smoothed spectrogram.
    N_order : int
        Number of neighbours on each side to compare against.
        Higher → only sharper, more isolated peaks survive.
        Default 8 (tighter than the original drb default of 6).
    peak_r : float, close to but below 1.0
        Tolerance factor. A neighbour must be less than
        peak_r * pixel_value for the pixel to survive.
        0.9993 means the neighbours must be at least 0.07% smaller —
        a very strict test that rejects broad, plateau-like features.

    Returns
    -------
    binary_map : 2D array, same shape as data_smoothed
        1.0 at local time-axis peaks, 0.0 everywhere else.
    """

    # Every pixel starts as a candidate (value = 1).
    # We will multiply by 0 wherever the local-peak condition fails.
    binary_map = np.ones_like(data_smoothed)

    # Pad N_order rows of zeros above and below so that comparisons near
    # the top and bottom edges of the array do not go out of bounds.
    pad_size     = N_order
    array_padded = np.pad(data_smoothed, ((pad_size, pad_size), (0, 0)))

    # ── Compare each pixel against neighbours at offsets 1 … N_order-1 ───────
    # For offset step (0-based, actual offset = step + 1):
    #
    #   pixel row      : array_padded[pad + step     : -pad + step     , :]
    #   neighbour BELOW: array_padded[pad + step + 1 : -pad + step + 1 , :]
    #   neighbour ABOVE: array_padded[pad - step - 1 : -pad - step - 1 , :]
    #
    # A pixel survives this step only if:
    #   peak_r * neighbour_below < pixel_value   (pixel > below neighbour)
    #   peak_r * neighbour_above < pixel_value   (pixel > above neighbour)
    #
    # Multiplying binary_map by the boolean result each step means
    # a pixel is zeroed out permanently the moment it fails any comparison.

    for step in range(pad_size - 1):

        pixel_row       = array_padded[pad_size + step    : -pad_size + step    , :]
        neighbour_below = array_padded[pad_size + step + 1: -pad_size + step + 1, :]
        neighbour_above = array_padded[pad_size - step - 1: -pad_size - step - 1, :]

        larger_than_below = (peak_r * neighbour_below < pixel_row)
        larger_than_above = (peak_r * neighbour_above < pixel_row)

        binary_map = binary_map * (larger_than_below & larger_than_above)

    return binary_map

def hough_detect(binary_map, dyspec,
                 threshold=30, line_gap=10, line_length=50,
                 theta=np.linspace(np.pi/2 - np.pi/8,
                                   np.pi/2 - 1/180*np.pi, 300)):
    """
    Find near-vertical line features in the binary spectrogram map.

    A Type III radio burst drifts from high to low frequency over time,
    producing a near-vertical streak in the binary map (time on x-axis,
    frequency on y-axis). The probabilistic Hough transform finds all
    such streaks that meet the minimum length and support requirements.

    Parameters
    ----------
    binary_map : 2D array
        Output of binarization() — 1 at candidate peak pixels, 0 elsewhere.
    dyspec : 2D array
        Raw spectrogram (passed for shape reference, not modified here).
    threshold : int
        Minimum number of pixels that must vote for a line.
        Lower → more lines found including more noise.
        Default 30 (tuned for OSRA f2 data).
    line_gap : int
        Maximum gap in pixels allowed within one line segment.
        Bridges small breaks in a burst drift track.
    line_length : int
        Minimum length in pixels a line must span to be reported.
    theta : 1D array (radians)
        Orientations to search. Default spans 22.5° to ~0.5° off vertical,
        covering realistic Type III drift slopes.
        Exactly vertical (90°) is excluded because it would imply
        an instantaneous (infinite) frequency drift — not physical.

    Returns
    -------
    lines : list of ((x0, y0), (x1, y1)) tuples
        Each entry is one detected line segment defined by its two
        endpoints in pixel coordinates
        (x = time index, y = frequency channel index).
    """

    lines = probabilistic_hough_line(
        binary_map,
        threshold=threshold,
        line_gap=line_gap,
        line_length=line_length,
        theta=theta
    )

    return lines


def point_to_line_distance(p1, p2, p3):
    """
    Compute the perpendicular distance from point p3 to the line
    defined by points p1 and p2.

    Uses the 2D cross-product formula:
        distance = |cross(p2 - p1, p1 - p3)| / norm(p2 - p1)

    In 2D the cross product is a scalar (the z-component of the
    3D cross product), giving the signed area of the parallelogram
    formed by the two vectors. Dividing by the line length gives
    the perpendicular height, i.e. the shortest distance from p3
    to the infinite line through p1 and p2.

    Parameters
    ----------
    p1, p2 : array-like, shape (2,)
        Two points defining the reference line.
    p3 : array-like, shape (2,)
        The point whose distance to the line is measured.

    Returns
    -------
    float : perpendicular distance in pixels.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    line_vector      = p2 - p1          # vector along the line
    point_vector     = p1 - p3          # vector from p3 to p1

    cross_product    = np.cross(line_vector, point_vector)   # scalar in 2D
    line_length      = np.linalg.norm(line_vector)

    distance = np.abs(cross_product) / line_length

    return distance

def line_grouping(lines, min_dist=3):
    """
    Cluster Hough line segments that belong to the same burst event.

    Lines are sorted by their starting time position (y-coordinate of
    the first endpoint), then examined pair by pair. Two consecutive
    lines are placed in the same group if at least one endpoint of the
    second line lies within min_dist pixels of the first line
    (perpendicular distance). Otherwise a new group is opened.

    Parameters
    ----------
    lines : list of ((x0,y0),(x1,y1)) tuples
        Detected Hough line segments from hough_detect().
    min_dist : float (pixels)
        Maximum perpendicular distance for two lines to be considered
        part of the same burst. Default 3 pixels.

    Returns
    -------
    line_groups : list of lists
        Each inner list contains all line segments assigned to one burst.
        Groups of exactly 1 line are single-line detections — likely noise.
        Groups of 2 or more lines define a drift track and are treated as
        real burst candidates by get_info_from_linegroupt_fits_cutout.
    """

    # ── Guard: return immediately if no lines were detected ───────────────────
    if len(lines) == 0:
        return []

    # ── Sort by the y-coordinate (time index) of each line's first endpoint ──
    lines_sorted = sorted(lines, key=lambda line: line[0][1])

    # ── Build groups iteratively ───────────────────────────────────────────────
    # Seed with the first line in its own group.
    line_groups = [[lines_sorted[0]]]

    for idx in range(len(lines_sorted) - 1):

        # Extract the four endpoints of the current and next line
        current_line            = lines_sorted[idx]
        next_line               = lines_sorted[idx + 1]
        point_A, point_B        = np.array(current_line[0]), np.array(current_line[1])
        point_C, point_D        = np.array(next_line[0]),    np.array(next_line[1])

        # Measure how close each endpoint of the next line is to the current line
        dist_C = point_to_line_distance(point_A, point_B, point_C)
        dist_D = point_to_line_distance(point_A, point_B, point_D)

        # If either endpoint is close enough → same burst, append to current group
        # Otherwise → new burst, open a fresh group
        if min(dist_C, dist_D) < min_dist:
            line_groups[-1].append(next_line)
        else:
            line_groups.append([next_line])

    return line_groups

def get_info_from_linegroupt_fits_cutout(line_sets, t_fits_cutout, f_fits):
    """
    Fit a frequency drift curve to each detected burst group and extract
    physical parameters.

    For each group of Hough lines (with more than 1 line):
      1. Map pixel coordinates back to real time (seconds) and frequency (MHz).
      2. Fit the Type III drift model  f(t) = rt.freq_drift_f_t  to the points.
      3. Store the fitted curve, exciter velocity v_beam, frequency range,
         and time range for plotting and cataloguing.

    Time axis is handled in seconds-since-window-start to avoid floating
    point precision problems with matplotlib date numbers.

    Parameters
    ----------
    line_sets     : list of lists — output of line_grouping()
    t_fits_cutout : 1D array of datetime objects for the current hour window
    f_fits        : 1D array of frequency values (MHz)

    Returns
    -------
    v_beam           : list of exciter velocities (fraction of c) per burst
    f_range_burst    : list of [f_min, f_max] per burst (MHz)
    t_range_burst    : list of [t_start, t_end] as datetime objects per burst
    model_curve_set  : list of [t_array, f_array] model curves as datetimes + MHz
    t_set_arr_set    : list of raw time arrays (seconds) used for fitting
    f_set_arr_set    : list of raw frequency arrays (MHz) used for fitting
    t_model_arr      : model time array from the last successfully fitted burst
    f_model_arr      : model frequency array from the last successfully fitted burst
    """

    # ── Build index arrays for interpolation ─────────────────────────────────
    # interp1d maps pixel row/column indices → real time/frequency values.
    t_idx_arr = np.arange(0, t_fits_cutout.shape[0])
    f_idx_arr = np.arange(0, f_fits.shape[0])

    # Convert t_fits_cutout to Python datetimes (safe for total_seconds())
    t_fits_cutout_dt = to_python_datetime(t_fits_cutout)

    # Convert datetimes to elapsed seconds from the start of the window.
    # This gives a clean numeric axis for curve_fit without large floating
    # point numbers (matplotlib date floats are days since 1970 ~ 19000,
    # which caused numerical instability in the original version).
    t_fits_cutout_num = np.array(
        [(t - t_fits_cutout_dt[0]).total_seconds() for t in t_fits_cutout_dt]
    )

    # Build interpolators: pixel index → real value
    t_interf = interpolate.interp1d(t_idx_arr, t_fits_cutout_num)  # → seconds
    f_interf = interpolate.interp1d(f_idx_arr, f_fits)              # → MHz

    # ── Initialise output lists ───────────────────────────────────────────────
    v_beam          = []
    f_range_burst   = []
    t_range_burst   = []
    model_curve_set = []
    t_set_arr_set   = []
    f_set_arr_set   = []
    t_set_arr       = []
    f_set_arr       = []
    t_model_arr     = []
    f_model_arr     = []

    # ── Process each burst group ──────────────────────────────────────────────
    for lines in line_sets:

        # Skip single-line groups — not enough points to define a drift track
        if len(lines) == 1:
            continue

        try:
            # Collect all pixel coordinates from every line in this group.
            # Each Hough line has two endpoints: (x0,y0) and (x1,y1)
            # where x = frequency channel index, y = time index.
            x_set = []   # time pixel indices
            y_set = []   # frequency pixel indices
            for line in lines:
                x_set.append(line[0][1])   # endpoint 1 time index
                x_set.append(line[1][1])   # endpoint 2 time index
                y_set.append(line[0][0])   # endpoint 1 freq index
                y_set.append(line[1][0])   # endpoint 2 freq index

            # Map pixel indices to real physical values
            t_set_arr = t_interf(x_set)   # seconds since window start
            f_set_arr = f_interf(y_set)   # MHz

            # ── Fit the Type III drift model ──────────────────────────────────
            # rt.freq_drift_f_t(t, v, t0) models the frequency as a function
            # of time for a beam travelling at speed v (fraction of c).
            # Initial guess: v=0.1c, t0 slightly before the earliest point.
            popt, pcov = optimize.curve_fit(
                rt.freq_drift_f_t,
                t_set_arr, f_set_arr,
                p0=(0.1, np.min(t_set_arr) - 3.0),   # t0 offset in seconds
                method="lm"
            )

            # ── Build the fitted model curve (50 points) ──────────────────────
            t_model_arr = np.linspace(
                rt.freq_drift_t_f(np.min(f_set_arr), *popt),
                rt.freq_drift_t_f(np.max(f_set_arr), *popt),
                50
            )
            f_model_arr = rt.freq_drift_f_t(t_model_arr, popt[0], popt[1])

            # Convert model time (seconds) back to Python datetime objects
            t_model_arr = np.array([
                t_fits_cutout_dt[0] + datetime.timedelta(seconds=float(s))
                for s in t_model_arr
            ])

            # ── Store results ─────────────────────────────────────────────────
            model_curve_set.append([t_model_arr, f_model_arr])

            # Start and end times of the burst as datetime objects
            t_range_burst.append([
                t_fits_cutout_dt[0] + datetime.timedelta(
                    seconds=float(rt.freq_drift_t_f(np.min(f_set_arr), *popt)[0])),
                t_fits_cutout_dt[0] + datetime.timedelta(
                    seconds=float(rt.freq_drift_t_f(np.max(f_set_arr), *popt)[0]))
            ])

            f_range_burst.append([np.min(f_set_arr), np.max(f_set_arr)])
            v_beam.append(popt[0])           # exciter speed as fraction of c
            t_set_arr_set.append(t_set_arr)
            f_set_arr_set.append(f_set_arr)

        except Exception:
            # curve_fit failed for this group (e.g. degenerate geometry,
            # too few unique points) — skip silently and continue.
            pass

    return (v_beam, f_range_burst, t_range_burst, model_curve_set,
            t_set_arr_set, f_set_arr_set, t_model_arr, f_model_arr)
    
def get_info_from_linegroup(line_sets, t_fits, f_fits):
    """
    Same purpose as get_info_from_linegroupt_fits_cutout but operates on
    the FULL file time array rather than an hourly cutout.

    The time axis here is expressed as matplotlib date floats (days since
    the epoch), which is why the conversion formula differs:
        elapsed_seconds = (t_float - min_t_float) * 24 * 3600

    Use this function when line_sets was built from the full t_fits array.
    Use get_info_from_linegroupt_fits_cutout when working with hourly slices.

    Parameters and returns are identical to get_info_from_linegroupt_fits_cutout
    except t_fits and t_range_burst values are matplotlib date floats, not datetimes.
    """

    t_idx_arr = np.arange(0, t_fits.shape[0])
    f_idx_arr = np.arange(0, f_fits.shape[0])
    t_interf  = interpolate.interp1d(t_idx_arr, t_fits)   # → matplotlib date float
    f_interf  = interpolate.interp1d(f_idx_arr, f_fits)   # → MHz

    v_beam          = []
    f_range_burst   = []
    t_range_burst   = []
    model_curve_set = []
    t_set_arr_set   = []
    f_set_arr_set   = []
    t_set_arr       = []
    f_set_arr       = []
    t_model_arr     = []
    f_model_arr     = []

    for lines in line_sets:

        if len(lines) == 1:
            continue

        try:
            x_set = []
            y_set = []
            for line in lines:
                x_set.append(line[0][1])
                x_set.append(line[1][1])
                y_set.append(line[0][0])
                y_set.append(line[1][0])

            # Convert matplotlib date float differences to seconds
            t_set_arr = (t_interf(x_set) - np.min(t_fits)) * 24 * 3600
            f_set_arr = f_interf(y_set)

            popt, pcov = optimize.curve_fit(
                rt.freq_drift_f_t,
                t_set_arr, f_set_arr,
                p0=(0.1, np.min(t_set_arr) - 3. / 3600 / 24),
                method="lm"
            )

            t_model_arr = np.linspace(
                rt.freq_drift_t_f(np.min(f_set_arr), *popt),
                rt.freq_drift_t_f(np.max(f_set_arr), *popt),
                50
            )
            f_model_arr = rt.freq_drift_f_t(t_model_arr, popt[0], popt[1])

            # Convert seconds back to matplotlib date float
            t_model_arr = t_model_arr / (24 * 3600) + np.min(t_fits)

            model_curve_set.append([t_model_arr, f_model_arr])
            t_range_burst.append([
                rt.freq_drift_t_f(np.min(f_set_arr), *popt)[0] / (24*3600) +                           np.min(t_fits),
                rt.freq_drift_t_f(np.max(f_set_arr), *popt)[0] / (24*3600) + np.min(t_fits)
            ])
            f_range_burst.append([np.min(f_set_arr), np.max(f_set_arr)])
            v_beam.append(popt[0])
            t_set_arr_set.append(t_set_arr)
            f_set_arr_set.append(f_set_arr)

        except Exception:
            pass

    return (v_beam, f_range_burst, t_range_burst, model_curve_set,
            t_set_arr_set, f_set_arr_set, t_model_arr, f_model_arr)    
    
def append_daily_csv(hourly_results, date,
                     output_dir="outputs/solar_cycle"):
    """
    Append one day's detection results as a SINGLE ROW to that year's CSV.

    WHY ONE ROW PER DAY?
    --------------------
    The previous layout wrote 24 rows per day (one per hour) plus a TOTAL
    row, making the file 25x longer than necessary and requiring a groupby
    to recover any day-level statistic.  The new layout stores every piece
    of information for one day in one row:

        date  | total_bursts | total_raw_groups | total_samples
              | bursts_h00 ... bursts_h23        (24 columns)
              | raw_h00    ... raw_h23            (24 columns)
              | samples_h00 ... samples_h23       (24 columns)

    Benefits of this layout
    -----------------------
    - One row per day — easy to read in a spreadsheet or with pd.read_csv()
      without any aggregation step.
    - Time-series analysis across the solar cycle just needs 'total_bursts' —
      no groupby required.
    - Hourly breakdown is still accessible: df['bursts_h06'] gives the
      06:00-07:00 count for every day in the file.
    - Vectorised pandas/numpy operations work naturally.

    HOW TO UPGRADE EXISTING LONG-FORMAT FILES
    ------------------------------------------
    If you already have a file in the old 25-row-per-day format, call
    the helper convert_long_to_wide_csv() once — it rewrites the file
    in the new layout without losing any data and backs up the original.

    Column naming convention
    ------------------------
    bursts_h00  ... bursts_h23   — confirmed burst count for each UTC hour
    raw_h00     ... raw_h23      — raw group count for each UTC hour
    samples_h00 ... samples_h23  — number of time records for each UTC hour

    File naming (unchanged from the previous version)
    --------------------------------------------------
    outputs/solar_cycle/bursts_YYYY.csv
    e.g. bursts_2003.csv, bursts_2004.csv ... bursts_2014.csv

    Duplicate-date guard
    --------------------
    If `date` already exists in the file (e.g. the notebook was re-run)
    the existing row is REPLACED rather than duplicated.  This makes the
    function idempotent — safe to call more than once for the same day.

    Parameters
    ----------
    hourly_results : list of 24 dicts, indexed 0-23
        Each dict must contain:
            'hour'       : int  0-23
            'bursts'     : int  confirmed burst count
            'raw_groups' : int  total line groups (including single-line)
            'samples'    : int  number of time samples in that hour
        Night or gap hours should already be set to 0 by the caller.

    date : str
        ISO date string, e.g. "2003-10-27".  Year is extracted from it
        to select the correct yearly output file.

    output_dir : str
        Directory for yearly CSV files.  Created if it does not exist.

    Returns
    -------
    str : absolute path of the yearly CSV file that was written to.
    """

    # ── Validate input length ─────────────────────────────────────────────────
    # The caller (hour loop) must have produced exactly 24 entries.
    # Catching this here gives a clear error message instead of a silent
    # misalignment between column names and values.
    if len(hourly_results) != 24:
        raise ValueError(
            f"append_daily_csv expected 24 hourly results, got {len(hourly_results)}. "
            "Make sure every branch of the hour loop calls hourly_results.append()."
        )

    # ── Sort entries by hour so columns are always in 00->23 order ───────────
    # Sorting is a safety measure in case the caller's loop ran out of order.
    entries = sorted(hourly_results, key=lambda e: e["hour"])

    # ── Build the 72 hourly value columns ────────────────────────────────────
    # Three groups of 24: bursts, raw_groups, samples.
    # Zero-padded hour tags (h00 ... h23) keep columns in sort order.
    burst_cols  = {f"bursts_h{h:02d}":  entries[h]["bursts"]     for h in range(24)}
    raw_cols    = {f"raw_h{h:02d}":     entries[h]["raw_groups"] for h in range(24)}
    sample_cols = {f"samples_h{h:02d}": entries[h]["samples"]    for h in range(24)}

    # ── Compute daily totals ──────────────────────────────────────────────────
    total_bursts  = sum(e["bursts"]     for e in entries)
    total_raw     = sum(e["raw_groups"] for e in entries)
    total_samples = sum(e["samples"]    for e in entries)

    # ── Assemble the single row as a dict ─────────────────────────────────────
    # Column order: date | totals | hourly bursts | hourly raw | hourly samples
    row = {
        "date"             : date,
        "total_bursts"     : total_bursts,
        "total_raw_groups" : total_raw,
        "total_samples"    : total_samples,
        **burst_cols,    # bursts_h00 ... bursts_h23
        **raw_cols,      # raw_h00    ... raw_h23
        **sample_cols,   # samples_h00 ... samples_h23
    }

    new_row_df = pd.DataFrame([row])   # single-row DataFrame

    # ── Select the correct yearly file ────────────────────────────────────────
    year     = date.split("-")[0]                         # e.g. "2003"
    filename = f"bursts_{year}.csv"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    # ── Write or update ───────────────────────────────────────────────────────
    if not os.path.exists(filepath):
        # Brand-new file — write with header.
        new_row_df.to_csv(filepath, mode="w", header=True, index=False)
        action = "created"

    else:
        # File already exists — load it and check for a duplicate date.
        existing_df = pd.read_csv(filepath, dtype={"date": str})

        if date in existing_df["date"].values:
            # Replace the existing row for this date (idempotent re-run).
            existing_df = existing_df[existing_df["date"] != date]
            updated_df  = pd.concat([existing_df, new_row_df], ignore_index=True)
            updated_df.sort_values("date", inplace=True)
            updated_df.to_csv(filepath, mode="w", header=True, index=False)
            action = "replaced"
        else:
            # New date — append a single row, no header needed.
            new_row_df.to_csv(filepath, mode="a", header=False, index=False)
            action = "appended"

    print(
        f"  [{date}] {action} -> {filepath}  "
        f"(total bursts={total_bursts}, "
        f"raw_groups={total_raw}, "
        f"samples={total_samples})"
    )

    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the whole-solar-cycle summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def build_solar_cycle_csv(output_dir="outputs/solar_cycle",
                          cycle_filename="bursts_solar_cycle.csv",
                          years=None):
    """
    Merge all yearly CSV files into one solar-cycle-wide CSV.

    The yearly files (bursts_YYYY.csv) are left UNCHANGED.  This function
    reads them, concatenates the rows in chronological order, and writes a
    separate summary file.  Re-running it rebuilds the summary from the
    current state of the yearly files — it is fully idempotent.

    When to call this
    -----------------
    - After processing the last day of a year (or the whole cycle).
    - Whenever you want an up-to-date whole-cycle view for analysis.
    - After back-filling missing days from earlier years.

    The cycle file is NOT updated automatically when append_daily_csv is
    called, because rebuilding it on every single day-write would be slow
    for large cycles.  Call this function explicitly when you need it.

    Output columns
    --------------
    Identical to the per-year files, plus one extra column after 'date':
        year  — integer calendar year, useful for year-level groupby

    Parameters
    ----------
    output_dir : str
        Directory that contains the yearly bursts_YYYY.csv files.
        The cycle file is also written here.

    cycle_filename : str
        Name of the output whole-cycle CSV file.
        Default "bursts_solar_cycle.csv" — safe name that never clashes
        with the "bursts_YYYY.csv" pattern.

    years : list of int or None
        Explicit list of years to include, e.g. [2003, 2004, 2005].
        If None, every bursts_YYYY.csv found in output_dir is included
        automatically.  Use an explicit list to exclude partially-processed
        years from the cycle summary.

    Returns
    -------
    str  : path to the written cycle CSV,  or  None if no data was found.

    Usage example
    -------------
    # After processing all days of all years:
    cycle_path = build_solar_cycle_csv(output_dir="outputs/solar_cycle")
    df_cycle   = pd.read_csv(cycle_path, parse_dates=["date"])

    # Daily burst time series across the whole cycle:
    df_cycle.plot(x="date", y="total_bursts")

    # Mean burst count per year:
    df_cycle.groupby("year")["total_bursts"].mean()

    # Retrieve the 12:00 burst column for every day in 2003:
    df_2003 = df_cycle[df_cycle["year"] == 2003]["bursts_h12"]
    """

    cycle_path = os.path.join(output_dir, cycle_filename)

    # ── Discover yearly files ─────────────────────────────────────────────────
    if years is not None:
        # Caller specified which years to include.
        yearly_files = [
            os.path.join(output_dir, f"bursts_{y}.csv")
            for y in sorted(years)
        ]
        for fp in yearly_files:
            if not os.path.exists(fp):
                print(f"  WARNING: {fp} not found — skipping.")
        yearly_files = [fp for fp in yearly_files if os.path.exists(fp)]

    else:
        # Auto-discover: match every bursts_YYYY.csv in output_dir,
        # but skip the cycle summary file itself.
        import glob
        pattern      = os.path.join(output_dir, "bursts_[0-9][0-9][0-9][0-9].csv")
        yearly_files = sorted(glob.glob(pattern))

    if not yearly_files:
        print(f"  No yearly CSV files found in '{output_dir}'. "
              "Run append_daily_csv first.")
        return None

    # ── Load and concatenate all yearly DataFrames ────────────────────────────
    frames = []
    for fp in yearly_files:
        try:
            df_year = pd.read_csv(fp, dtype={"date": str})
            frames.append(df_year)
            print(f"  Loaded {len(df_year):>4} days from {os.path.basename(fp)}")
        except Exception as exc:
            print(f"  WARNING: could not read {fp} — {exc}")

    if not frames:
        print("  No data loaded. Cycle CSV was not written.")
        return None

    df_cycle = pd.concat(frames, ignore_index=True)

    # ── Add a 'year' column immediately after 'date' ───────────────────────────
    # This makes year-level groupby trivial without parsing date strings.
    df_cycle.insert(1, "year", df_cycle["date"].str[:4].astype(int))

    # ── Sort chronologically and drop exact duplicate dates ───────────────────
    # Duplicates can appear if a year file was accidentally written twice.
    df_cycle.sort_values("date", inplace=True)
    df_cycle.drop_duplicates(subset="date", keep="last", inplace=True)
    df_cycle.reset_index(drop=True, inplace=True)

    # ── Write the cycle file ──────────────────────────────────────────────────
    df_cycle.to_csv(cycle_path, index=False)

    print(
        f"\n  Solar-cycle CSV written -> {cycle_path}\n"
        f"  Total days : {len(df_cycle)}\n"
        f"  Date range : {df_cycle['date'].iloc[0]}  ->  {df_cycle['date'].iloc[-1]}\n"
        f"  Years      : {sorted(df_cycle['year'].unique().tolist())}"
    )

    return cycle_path


# ─────────────────────────────────────────────────────────────────────────────
# Migration helper: convert old long-format files to the new wide format
# ─────────────────────────────────────────────────────────────────────────────

def convert_long_to_wide_csv(filepath, output_dir=None):
    """
    One-time migration: convert a legacy long-format CSV to the new wide format.

    The old format had 25 rows per day (24 hourly + 1 TOTAL row).
    The new format has 1 row per day with 75 columns.

    The original file is backed up to <name>_legacy.csv before the
    converted version is written in its place.  Nothing is deleted.

    Parameters
    ----------
    filepath   : str  Path to the old-format CSV (e.g. "outputs/.../bursts_2003.csv")
    output_dir : str  Where to write the converted file.  Defaults to the
                      same directory as the input file.

    Returns
    -------
    str : path of the new wide-format CSV.
    """
    import shutil

    df_old = pd.read_csv(filepath, dtype={"date": str, "hour": str})

    # ── Check this is actually a long-format file ─────────────────────────────
    if "hour" not in df_old.columns:
        print(f"  {filepath} does not look like a legacy long-format file — skipping.")
        return filepath

    # ── Keep only the 24 hourly rows; drop TOTAL rows ─────────────────────────
    df_hourly = df_old[df_old["hour"] != "TOTAL"].copy()
    df_hourly["hour_int"] = df_hourly["hour"].str[:2].astype(int)

    wide_rows = []
    for date, group in df_hourly.groupby("date"):
        group     = group.sort_values("hour_int")
        hour_data = {row["hour_int"]: row for _, row in group.iterrows()}

        # Fill missing hours with zeros so every day has exactly 24 entries
        entries = []
        for h in range(24):
            if h in hour_data:
                entries.append(hour_data[h])
            else:
                entries.append({"bursts": 0, "raw_groups": 0, "samples": 0})

        burst_cols  = {f"bursts_h{h:02d}":  entries[h]["bursts"]     for h in range(24)}
        raw_cols    = {f"raw_h{h:02d}":     entries[h]["raw_groups"] for h in range(24)}
        sample_cols = {f"samples_h{h:02d}": entries[h]["samples"]    for h in range(24)}

        wide_rows.append({
            "date"             : date,
            "total_bursts"     : sum(e["bursts"]     for e in entries),
            "total_raw_groups" : sum(e["raw_groups"] for e in entries),
            "total_samples"    : sum(e["samples"]    for e in entries),
            **burst_cols, **raw_cols, **sample_cols,
        })

    df_wide = pd.DataFrame(wide_rows).sort_values("date").reset_index(drop=True)

    # ── Back up the original file ─────────────────────────────────────────────
    backup_path = filepath.replace(".csv", "_legacy.csv")
    shutil.copy2(filepath, backup_path)
    print(f"  Original backed up -> {backup_path}")

    # ── Write the new wide-format file ────────────────────────────────────────
    out_dir  = output_dir or os.path.dirname(filepath)
    out_path = os.path.join(out_dir, os.path.basename(filepath))
    df_wide.to_csv(out_path, index=False)
    print(f"  Wide-format written -> {out_path}  ({len(df_wide)} days)")

    return out_path


def append_into_json(old_json, v_beam, f_range_burst, t_range_burst):
    """
    Add Type III burst detection results to an existing JSON structure.

    Designed to be called once per time window after
    get_info_from_linegroupt_fits_cutout has been run. The results are
    inserted under the key 'event' in old_json and the updated dict
    is returned.

    Parameters
    ----------
    old_json      : dict  — existing JSON/dict to update (e.g. metadata for one file)
    v_beam        : list  — exciter velocities (fraction of c) per burst
    f_range_burst : list  — [f_min, f_max] in MHz per burst
    t_range_burst : list  — [t_start, t_end] as datetime objects per burst

    Returns
    -------
    old_json : dict with 'event' key added or overwritten.

    Output structure
    ----------------
    old_json['event'] = {
        'detection' : True,
        'type'      : 'III',
        'detail'    : [
            {
                'v_beam'     : float,
                'freq_range' : [f_min, f_max],
                'time_range' : [t_start, t_end],
                'str_time'   : "HH:MM:SS"   ← burst start time as string
            },
            ...   one dict per detected burst
        ]
    }
    """

    # Build one detail dict per detected burst
    event_detail = []
    for idx, v_cur in enumerate(v_beam):
        event_detail.append({
            'v_beam'     : v_cur,
            'freq_range' : f_range_burst[idx],
            'time_range' : t_range_burst[idx],
            # Format the burst start time as HH:MM:SS for quick readability.
            # t_range_burst[idx][0] is a Python datetime object.
            'str_time'   : t_range_burst[idx][0].strftime("%H:%M:%S")
        })

    # Insert the event block into the existing JSON structure
    old_json['event'] = {
        'detection' : True,
        'type'      : 'III',
        'detail'    : event_detail
    }

    return old_json