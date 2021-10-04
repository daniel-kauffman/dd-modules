# Speaker Diarization Library

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os
import re
import subprocess
from os import path

import pandas as pd


def diarize(wav_path, srt_path = None, dsrt_path = None):
    """
    Run the LIUM diarization program on an audio file.
    
    Args:
        wav_path: A str representing the path of a .wav file.
        srt_path: A str representing the path to output a .srt file.
        dsrt_path: A str representing the path to output a diarized .dsrt file.
    
    Returns:
        A DataFrame containing diarized segment data.
    
    Raises:
        IOError: LIUM diarization JAR not found.
    """
    file_id, _ = os.path.splitext(os.path.basename(wav_path))
    if srt_path is None or not os.path.exists(srt_path):
        if srt_path is None:
            srt_path = file_id + ".srt"
        srt_path = download_srt(file_id, srt_path = srt_path)
    if dsrt_path is None:
        dsrt_path = os.path.join(os.path.dirname(srt_path), file_id + ".dsrt")
    seg_path = os.path.join(os.path.dirname(dsrt_path), file_id + ".seg")
    if not os.path.exists(seg_path):
        module_path = path.dirname(path.abspath(__file__)) + "/"
        pattern = re.compile("lium_spkdiarization-\d+.\d+.\d+.jar")
        jar_paths = sorted([file_path for file_path
                            in os.listdir(module_path)
                                      if re.match(pattern, file_path)],
                           reverse = True)
        if len(jar_paths) == 0:
            raise IOError("LIUM Not Found")
        command = ["/usr/bin/java", "-Xmx1024m", "-jar", module_path+jar_paths[0], 
                   "--fInputMask=" + wav_path, "--sOutputMask=" + seg_path, 
                   "--doCEClustering", file_id.strip("-")]
        with open(os.devnull, "w") as null:
            exit_code = subprocess.call(command, stdout = null, stderr = null)
    if os.path.exists(seg_path):
        if file_id[0] == "-":
            command = ["sed", "-i", "s/{0}/-{0}/".format(file_id[1:]),
                       seg_path]
            subprocess.call(command)
        seg_df = seg_to_df(seg_path)
        srt_df = srt_to_df(srt_path)
        srt_df = label_srt_text(srt_df, seg_df)
        return write_srt(srt_df, srt_path = dsrt_path)
    else:
        print("exit_code=%s. seg_path=%s does not exist!" % (exit_code, seg_path))

def download_srt(file_id, srt_path = None):
    """
    Download a transcribed .srt file from Digital Democracy's deployment server.
    
    Args:
        file_id: A str representing a File ID.
        srt_path: A str representing the path to download a .srt file.
    
    Returns:
        The path to the downloaded .srt file.
    
    Raises:
        IOError: Unable to download SRT.
    """
    if srt_path is None:
        srt_path = file_id + ".srt"
    url = ""
    if not os.path.exists(srt_path):
        command = ["wget", "-O", srt_path, url.format(file_id)]
        with open(os.devnull) as null:
            subprocess.call(command, stdout = null, stderr = null)
    if os.path.exists(srt_path):
        if os.path.getsize(srt_path) > 0:
            return srt_path
        os.remove(srt_path)
    raise IOError("Error Downloading SRT:", srt_path)


def seg_to_df(seg_path):
    """
    Parse a .seg file and store the segments in a DataFrame.
    
    Args:
        seg_path: A str representing the path to a .seg file.
    
    Returns:
        A DataFrame with the following schema:
            file_id    object
            start       int64
            end         int64
            label      object
    """
    file_id, _ = os.path.splitext(os.path.basename(seg_path))
    fmt_str = "{0} 1 (\d+) (\d+) [F|M|U] [S|T] U (S\d+)\n"
    pattern = re.compile(fmt_str.format(file_id))
    with open(seg_path, "r") as fp:
        lines = fp.readlines()
    seg_list = []
    for line in lines:
        match = re.match(pattern, line)
        if match:
            start = int(match.group(1)) * 10
            end = start + int(match.group(2)) * 10
            label = match.group(3)
            seg_list.append({"file_id": file_id, "start": start, "end": end,
                             "label": label})
    seg_list = sorted(seg_list, key = lambda seg: seg["start"])
    columns = ["file_id", "start", "end", "label"]
    seg_df = pd.DataFrame(seg_list, index = range(1, len(seg_list) + 1),
                          columns = columns).rename_axis("segment")
    return reset_labels(seg_df)


def reset_labels(seg_df):
    """
    Relabel segments in numeric order (starting with S00) and ensure each
    label is three characters in length.
    
    Args:
        seg_df: A DataFrame containing diarized segment data.
            file_id    object
            start       int64
            end         int64
            label      object
    
    Returns:
        A DataFrame with the same schema as the input but with modified labels.
    """
    for i, label in enumerate(seg_df.label.unique()):
        n = 0 if label == "S0" else i + 1
        seg_df.loc[seg_df.label == label, "label"] = "S{0:02d}".format(n)
    return seg_df


def srt_to_df(srt_path, ms_delimit = ","):
    """
    Parse a .srt file and store the segments in a DataFrame.
    
    Args:
        srt_path: A str representing the path to a .srt file.
        ms_delmit: The str used to separate seconds and milliseconds in the
                   time code (defaults to "." but may sometimes be ",").
    
    Returns:
        A DataFrame with the following schema:
            file_id    object
            start       int64
            end         int64
            text      object
    
    Raises:
        IOError: The .srt file does not exist.
    """
    file_id, _ = os.path.splitext(os.path.basename(srt_path))
    if not os.path.exists(srt_path):
        srt_path = download_srt(file_id, srt_path = srt_path)
    if srt_path is None:
        raise IOError("File Not Found:", file_id)
    with open(srt_path, "r") as fp:
        lines = fp.readlines()
    reg_ex = (r"(\d{{2}}:\d{{2}}:\d{{2}}{0}\d{{3}}) --> " +
              r"(\d{{2}}:\d{{2}}:\d{{2}}{0}\d{{3}})").format(ms_delimit)
    pattern = re.compile(reg_ex)
    rows = []
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            start = get_milliseconds(match.group(1), ms_delimit = ms_delimit)
            start = align_milliseconds(start, nearest = 200)
            end = get_milliseconds(match.group(2), ms_delimit = ms_delimit)
            end = align_milliseconds(end, nearest = 200)
            text = lines[i + 1].strip()
            rows.append((file_id, start, end, text))

    # adjust end time in case timeframes overlap
    for i in reversed(range(len(rows))):
        # end time is later than next start time
        if i > 0 and rows[i][1] < rows[i-1][2]:
            rows[i-1] = (rows[i-1][0], rows[i-1][1], rows[i][1], rows[i-1][3]) 
         
    columns = ["file_id", "start", "end", "text"]
    segments = pd.DataFrame(rows, index = range(1, len(rows) + 1),
                            columns = columns)
    return segments.rename_axis("segment")


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


def label_srt_text(srt_df, seg_df):
    """
    Add diarization segment labels to each of the given transcribed utterances.
    
    Args:
        srt_df: A DataFrame containing transcribed utterance data.
            file_id    object
            start       int64
            end         int64
            text       object
        seg_df: A DataFrame containing diarized segment data.
            file_id    object
            start       int64
            end         int64
            label      object
    
    Returns:
        A DataFrame with the following schema:
            file_id    object
            start       int64
            end         int64
            text       object
            label      object
    """
    srt_seconds = expand_segments(srt_df, step = 200)
    seg_seconds = expand_segments(seg_df, step = 200)[["label"]]
    seconds = srt_seconds.join(seg_seconds)

    labels = {s:find_mode(group[(group != "NaN") & (group != "S00")], default = "S00")
              for s, group in seconds.groupby("segment").label}
    add_label_column(srt_df, labels)
    return srt_df

def add_label_column(srt_df, labels):
    """Add label column to the pandas dataframe (side-effect) and
    assign label values to the column.

    Args:
        srt_df: A DataFrame containing transcribed utterance data.
        labels: A dictionary of segment => label
    """
    srt_df["label"] = "S00" 
    for seg, row in srt_df.iterrows():
        if seg in labels:
            srt_df.at[seg, "label"] = labels[seg]

def expand_segments(segments, step = 1000):
    """
    Create a DataFrame from utterance data such that each row represents
    one interval (e.g. second) of an utterance.
    
    Args:
        segments: A DataFrame containing utterance data.
        step: An integer specifying the length (in ms) of each interval.
    
    Returns:
        A DataFrame of utterance data split into seconds.
    """
    segments = segments.copy().reset_index()
    data = []
    time_codes = []
    for _, row in segments.iterrows():
        start = align_milliseconds(row.start, nearest = step, use_round = False)
        end = align_milliseconds(row.end, nearest = step, use_round = False)
        for millisecond in range(start, end, step):
            time_codes.append(millisecond)
            data.append(row)
    seconds = pd.DataFrame(data, index = time_codes).rename_axis("second")
    return seconds[~seconds.index.duplicated(keep = "first")]


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


def write_srt(srt_df, srt_path = None):
    """
    Write segment data from a DataFrame to a .srt file.
    
    Args:
        srt_df: A DataFrame containing segment data.
        srt_path: A str representing the path to write a .srt file.
    
    Returns:
        A str representing the path to the new .srt file.
    """
    if srt_path is None:
        srt_path = srt_df.iloc[0].file_id + ".srt"
    with open(srt_path, "w") as fp:
        for i, row in srt_df.iterrows():
            start_code = get_time_code(row.start)
            end_code = get_time_code(row.end)
            time_code = "{0} --> {1}".format(start_code, end_code)
            text = "{0}".format(row.label)
            fp.write("\n".join((str(i), time_code, text)) + "\n\n")
    if os.path.exists(srt_path):
        return srt_path


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
