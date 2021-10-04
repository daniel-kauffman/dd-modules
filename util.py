# Common Utility Library

from __future__ import division, print_function
import glob
import multiprocessing as mp
import os
import re
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

import common.db as db


def download_video_from_s3(file_id, verbose = False):
    """
    Download a video from Digital Democracy's Amazon S3 account.
    
    Args:
        file_id: A str representing a File ID.
    
    Returns:
        A str representing a path to the downloaded video.
    """
    vid_name = file_id + ".mp4"
    vid_path = get_dir("temp") + vid_name    
    if os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
        return vid_path
    else:
        s3_bucket = os.getenv("S3_VIDEO_BUCKET", "videostorage-us-west")
        url = "https://s3-us-west-2.amazonaws.com/" + \
              "{0}/videos/{1}/{2}".format(s3_bucket, file_id, vid_name)
        command = ['wget', '-O', vid_path, url]
        if verbose:
            print(" ".join(command))
        wget_proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        out, err = wget_proc.communicate()
        if wget_proc.returncode != 0:
            fmt_str = "ERROR: wget return code {0}"
            print(fmt_str.format(wget_proc.returncode))
            print(out)
        elif os.path.exists(vid_path) and os.path.getsize(vid_path) > 0:
            return vid_path


def convert_to_wav(vid_path, verbose = False):
    """
    Convert a video to the .wav audio format.
    
    Args:
        vid_path: A str representing the path to the video.
    
    Returns:
        A str representing the path to the .wav file.
    """
    wav_path = os.path.splitext(vid_path)[0] + ".wav"
    if not os.path.exists(wav_path):
        command = ["ffmpeg", "-i", vid_path, "-vn", "-f", "wav",
                   "-ar", "16000", "-ac", "1", wav_path]
        if verbose:
            print(" ".join(command))
        ffmpeg_proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr = subprocess.STDOUT)
        out, err = ffmpeg_proc.communicate()
        if ffmpeg_proc.returncode != 0:
            fmt_str = "ERROR: ffmpeg return code {0}"
            print(fmt_str.format(ffmpeg_proc.returncode))
            print(out)
    if os.path.exists(wav_path):
        return wav_path


def normalize_series(series, robust = True, use_zero = True):
    """
    Normalize the values in a Series to be between 0 and 1 such that all values
    in the Series add up to 1.
    
    Args:
        series: A Series with numerical values.
        robust: A bool that determines whether to ensure outliers do not
                drastically skew normalized values.
        use_zero: A bool indicating whether 0 should be used as the minimum
                  value when normalizing.
    
    Returns:
        A Series with values between 0 and 1 that add up to 1.
    """
    series = series.copy()
    if robust:
        z_scores = (series - series.mean()) / series.std()
        outliers = z_scores[z_scores < -1.0].index
        normal = series[~series.index.isin(outliers)]
        normal = normal + abs(normal.min()) + 1
        series[outliers] = 0.0
        series[normal.index] = normal
    s_min = 0.0 if use_zero else series.min()
    s_max = series.max()
    if s_min < s_max:
        series = (series - s_min) / (s_max - s_min)
    return series / series.sum()


def find_mode(series, default = None):
    """
    Retrieve the most common value in a series. If more than one value has the
    same frequency, the lowest-indexed value is chosen.
    
    Args:
        series: A Series of any comparable data type.
        default: The value to use if the given series is empty.
        
    Returns:
        The most common value from the series if non-empty; otherwise returns
        the given default value.
    """
    return (default if len(series.value_counts().index) == 0
                    else series.value_counts().index[0])


def get_root_name(path):
    """
    Extract a file name stripped of directories and extensions.
    
    Args:
        path: A str representing the path to a file.
    
    Returns:
        A str containing only the root name of the file.
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_vid_len(path):
    """
    Retrieve the length of a video in milliseconds.
    
    Args:
        path: A str representing the path to a video.
    
    Returns:
        An int specifying the number of milliseconds in the video.
    """
    command = ["ffmpeg", "-i", path]
    proc = subprocess.Popen(command, stdout = subprocess.PIPE,
                            stderr = subprocess.PIPE)
    stdout, stderr = proc.communicate()
    pat_str = r"Duration: (\d{2}:\d{2}:\d{2}.\d{2})"
    time_code = re.search(pat_str, stderr).group(1) + "0"   # convert to ms
    return get_milliseconds(time_code)


def get_dir(dir_key, state = None):
    """
    Get the path to a directory on disk.
    
    Args:
        dir_key: A str indicating the name of a directory.
        state: A two-character str representing a U.S. state abbreviation.
    
    Returns:
        A str containing an absolute directory path.
    
    Raises:
        ValueError: dir_key is not a valid directory name.
        ValueError: state is not a valid two-character U.S. state abbreviation.
    """
    dirs = {"pid": "/audio/pid/", "vid": "/audio/vid/",
            "srt": "/audio/srt/", "cielo-srt": "/audio/cielo-srt/",
            "seg": "/audio/seg/", "gmm": "/audio/gmm/",
            "jar": "/audio/jar/", "temp": "/audio/temp/"}
    if dir_key not in dirs.keys():
        raise ValueError("Invalid Directory Name")
    dir_name = dirs[dir_key]
    if state is not None:
        if len(state) != 2 or not state.isalpha():
            raise ValueError("Invalid U.S. State Abbreviation:", state)
        dir_name += state.lower() + "/"
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            traceback.print_exc()
    return dir_name


def get_milliseconds(time_code, ms_delimit = "."):
    """
    Extract the number of milliseconds from a time-coded string. 
    
    Args:
        time_code: A str containing a time code of the format HH:MM:SS.MS.
        ms_delmit: The str used to separate seconds and milliseconds in the
                   time code (defaults to "." but may sometimes be ",").
    
    Returns:
        An int specifying the number of milliseconds in the time code.
    """
    time_code, milliseconds = time_code.split(ms_delimit)
    split_code = time_code.split(":")
    seconds = ((int(split_code[0]) * 3600) + (int(split_code[1]) * 60) + 
               int(split_code[2]))
    return (seconds * 1000) + int(milliseconds)


def get_time_code(milliseconds, ms_delimit = "."):
    """
    Create a time code of the format HH:MM:SS.MS from a given number of
    milliseconds.
    
    Args:
        milliseconds: An int specifying the number of milliseconds to encode.  
        ms_delmit: The str to be used to separate seconds and milliseconds in
                   the time code (defaults to "." but may sometimes be ",").
    
    Returns:
        A str containing a time code of the format HH:MM:SS.MS.
    """
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    fmt_str = "{0:02d}:{1:02d}:{2:02d}{3}{4:03d}"
    return fmt_str.format(hours, minutes, seconds, ms_delimit, milliseconds)


def expand_utterances(utterances, step = 1000):
    """
    Create a DataFrame from utterance data such that each row represents
    one second of an utterance.
    
    Args:
        utterances: A DataFrame containing utterance data.
    
    Returns:
        A DataFrame of utterance data split into seconds.
    """
    utterances = utterances.copy().reset_index()
    data = []
    time_codes = []
    for _, row in utterances.iterrows():
        start = align_milliseconds(row.start, nearest = step, use_round = False)
        end = align_milliseconds(row.end, nearest = step, use_round = False)
        for millisecond in range(start, end, step):
            time_codes.append(millisecond)
            data.append(row)
    seconds = pd.DataFrame(data, index = time_codes).rename_axis("second")
    return seconds[~seconds.index.duplicated(keep = "first")]


def align_milliseconds(milliseconds, nearest = 1000, use_round = True):
    """
    Round milliseconds to a specified value.
    
    Args:
        milliseconds: An int of milliseconds to be aligned.  
        nearest: An int specifying the precision of the alignment (defaults to
                 1000, aligning milliseconds to the nearest thousand).
        use_round: A bool that causes alignment to round up or down if True
                   and always round down if False.
    
    Returns:
        An int of the rounded milliseconds.
    """
    if use_round:
        milliseconds = int(round(milliseconds / nearest) * nearest)
    else:
        milliseconds = milliseconds // nearest * nearest
    return milliseconds


def rank_probas(probas, n_pred = 5):
    """
    Convert prediction probabilities into ordinal rankings per speaker.
    
    Args:
        probas: A DataFrame of speaker probabilities per cluster label.
        n_pred: An int specifying the number of top predictions to use.
    
    Returns:
        A DataFrame of prediction rankings per cluster label.
    """
    n_pred = min(n_pred, len(probas.columns))
    rankings = probas.rank(axis = 1, method = "min", ascending = False)
    rankings = (n_pred - (rankings - 1)).clip(lower = 0)
    return rankings / float(n_pred)


def score_rankings(rankings):
    """
    Calculate the score for each set of rankings per cluster label.
    
    Args:
        rankings: A DataFrame of prediction rankings per cluster label.
    
    Returns:
        A Series of speaker labels to integer scores.
    """
    scores = rankings.apply(lambda row: 0 if int(row.name) not in row.index
                                          else row[int(row.name)], axis = 1)
    return scores.fillna(value = 0).rename("score")


def calculate_fscore(precision, recall, beta = 1):
    """
    Compute the F-score based on the given precision and recall, weighted by
    the given beta value.
    
    Args:
        precision: A float between 0 and 1.
        recall: A float between 0 and 1.
        beta: A positive float used to weigh the F-score.
    
    Returns:
        A float between 0 and 1 representing a weighted F-score.
    """
    if precision == 0 and recall == 0:
        return 0
    return ((1.0 + beta ** 2.0) * precision * recall /
            (beta ** 2.0 * precision + recall))


def convert_proba_dict_to_dataframe(proba_dict, clf_name):
    """
    Transform a dictionary of classification tests to a DataFrame. This
    function is primarily intended to convert the output of a convolutional
    neural network used for face recognition into a standard format.
    
    Args:
        proba_dict: A dict with the following structure:
                        {vid: [(label, [(pid, conf), ...]), ...], ...}
        clf_name: A str containing the name of a classifier.
    
    Returns:
        A DataFrame indexed by file ID, label, and classifier with Person IDs
        as columns.
    """
    vids = sorted(proba_dict.keys())
    query = """
            SELECT DISTINCT vid, fileId AS file_id
            FROM Video
            WHERE vid IN ({0});
            """.format(", ".join(str(vid) for vid in vids))
    records = db.query_database(query, index_col = "vid", verbose = verbose)
    series_list = []
    for vid, file_id in records.iterrows():
        for label, conf_list in proba_dict[vid]:
            data = {"file_id": file_id, "label": int(label), "clf": clf_name}
            data.update({int(pid): float(conf) for pid, conf in conf_list})
            series_list.append(pd.Series(data))
    probas = pd.DataFrame(series_list).set_index(["file_id", "label", "clf"])
    pids = sorted([column for column in probas.columns if type(column) == int])
    return probas[pids].sort_index()


def get_most_recent_mod_date(paths):
    """
    Find the most recent modification date among all files from the given list
    of paths.
    
    Args:
        paths: A list of str, each representing a path to a file on disk.
    
    Returns:
        A NumPy datetime64 object with the most recent modification date of all
        files from the given list of file paths.
    """
    if not paths:
        return pd.to_datetime("today")
    os.stat_float_times(False)
    latest = 0
    for path in paths:
        mod_time = os.path.getmtime(path)
        if latest < mod_time:
            latest = mod_time
    return np.datetime64(latest, "s").astype("datetime64[D]")


def map_processes(function, sequence, n_procs = 8, callback = None):
    """
    Map a sequence of elements to a function, with each function call in a
    separate process.
    
    Args:
        function: A function of arity 1 over which elements are applied.
        sequence: The elements to be passed to the function.
        n_procs: An int specifying the number of asychronous processes.
        callback: A function of arity 1 over which results may be applied.
    
    Returns:
        A list containing the returned values of each function call.
    """
    pool = mp.Pool(min(n_procs, len(sequence)), maxtasksperchild = 1)
    result = pool.map_async(function, sequence,
                            chunksize = 1, callback = callback)
    pool.close()
    pool.join()
    return result.get()


def run_silently(command, verbose = False):
    """
    Supress output of a command.
    
    Args:
        command: A list of str containing the command and its arguments to run.
    """
    null = open(os.devnull, "w")
    if verbose:
        print(" ".join(command))
    subprocess.call(command, stdout = null, stderr = null)
    null.close()


def print_progress(title, stop):
    """
    Write a progress message to stdout, overwriting previous messages using a
    carriage return.
    
    Args:
        title: A str with the progress message to display.
        stop: An int for the upper limit to display for the progress count.
              A newline is only printed when this function is called stop times.
    """
    start = 1
    while True:
        sys.stdout.write("\r{0}: {1:d} / {2:d}".format(title, start, stop))
        sys.stdout.flush()
        if start >= stop:
            break
        start += 1
        yield
    sys.stdout.write("\n")
    sys.stdout.flush()
    yield


def clean(dir_path, extensions = ["*"], verbose = False):
    """
    Remove all files with a given extension in a given directory.
    
    Args:
        dir_path: A str representing a directory.
        extensions: A list of str of file extensions to remove.
    """
    patterns = [dir_path + "/*." + ext for ext in extensions]
    file_paths = []
    for pattern in patterns:
        file_paths += glob.glob(pattern)
    for file_path in file_paths:
        os.remove(file_path)
        if verbose:
            print("Removed", file_path)
