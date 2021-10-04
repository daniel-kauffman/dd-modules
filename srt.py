#!/usr/bin/python2.7
#
# SRT Management Library

from __future__ import division, print_function
import os
import re
import subprocess
import sys

import pandas as pd

import common.diarization as diarization
import common.util as util


def load_utterances(file_id, state, srt_text = None, verbose = False):
    """
    Parse a machine-transcribed SRT into a DataFrame, providing a cluster label
    to each utterance as obtained from diarization.
    
    Args:
        file_id: A str representing a File ID.
        state: A str containing the US state abbreviation for the video. 
        srt_text: A list of str containing lines from a .srt file.
    
    Returns:
        A DataFrame with the following schema:
            file_id    object
            start       int64
            end         int64
            text       object
            label      object
    """
    utterances = (read_srt_text(srt_text, file_id) if srt_text else
                  read_srt(file_id, verbose = verbose))
    segments = diarization.load_segments(file_id, state, verbose = verbose)
    return label_utterances(utterances, segments)


def download_srt(file_id, verbose = False):
    """
    Download a transcribed .srt file from Digital Democracy's deployment server.
    
    Args:
        file_id: A str representing a File ID.
    
    Returns:
        The path to the downloaded .srt file.
    """
    srt_path = util.get_dir("cielo-srt") + file_id + ".srt"
    url = "http://deployment.digitaldemocracy.org/videos/transcripts/{0}.srt"
    if not os.path.exists(srt_path):
        command = ["wget", "-O", srt_path, url.format(file_id)]
        util.run_silently(command, verbose = verbose)
    if os.path.exists(srt_path):
        if os.path.getsize(srt_path) > 0:
            return srt_path
        os.remove(srt_path)
    raise IOError("Error Downloading SRT:", srt_path)


def read_srt(srt_path, verbose = False):
    """
    Read a transcribed .srt file and store its segments in a DataFrame.
    
    Args:
        srt_path: A str representing the path of a .srt file.
    
    Returns:
        A DataFrame containing segment data from the .srt file.
    
    Raises:
        IOError: The .srt file does not exist.
    """
    file_id = util.get_root_name(srt_path)
    if not os.path.exists(srt_path):
        srt_path = download_srt(file_id, verbose = verbose)
    if srt_path is None:
        raise IOError("File Not Found:", file_id)
    with open(srt_path, "r") as fp:
        lines = fp.readlines()
    return read_srt_text(lines, file_id)


def read_srt_text(srt_text, file_id):
    """
    Parse strings containing segment data to be stored in a DataFrame.
    
    Args:
        srt_text: A list of str containing lines from a .srt file.
        file_id: A str representing a File ID.
    
    Returns:
        A DataFrame containing segment data from the .srt file.
    """
    reg_ex = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    pattern = re.compile(reg_ex)
    rows = []
    for i, line in enumerate(srt_text):
        match = re.search(pattern, line)
        if match:
            start = util.get_milliseconds(match.group(1), ",")
            start = util.align_milliseconds(start, nearest = 200)
            end = util.get_milliseconds(match.group(2), ",")
            end = util.align_milliseconds(end, nearest = 200)
            text = srt_text[i + 1].strip()
            rows.append((file_id, start, end, text))
    columns = ["file_id", "start", "end", "text"]
    utterances = pd.DataFrame(rows, index = range(1, len(rows) + 1),
                              columns = columns)
    return utterances.rename_axis("segment")


def label_utterances(utterances, segments):
    """
    Add diarization segment labels to each of the given transcribed utterances.
    
    Args:
        utterances: A DataFrame containing transcribed utterance data.
            file_id    object
            start       int64
            end         int64
            text       object
        segments: A DataFrame containing diarized segment data.
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
    utt_seconds = util.expand_utterances(utterances, step = 200)
    seg_seconds = util.expand_utterances(segments, step = 200)[["label"]]
    seconds = utt_seconds.join(seg_seconds)
    labels = [util.find_mode(group[group != "S00"], default = "S00")
              for _, group in seconds.groupby("segment").label]
    utterances["label"] = labels
    return edit_labels(utterances)


def edit_labels(utterances):
    """
    Modify diarization labels based on textual cues in each utterance.
    
    Args: 
        utterances: A DataFrame containing transcribed utterance data.
            file_id    object
            start       int64
            end         int64
            text       object
            label      object
    
    Returns:
        A DataFrame with the same schema as the input but with modified labels.
    """
    utterances = utterances.copy()
    utt_seconds = util.expand_utterances(utterances)
    transitions = utterances[utterances.text.str.match("^>>.*")].index.tolist()
    transitions.append(len(utterances) + 1) # ensure final transition is edited
    start = utterances.index[0]
    for i in transitions:
        indexes = range(start, i)
        labels = utt_seconds[utt_seconds.segment.isin(indexes)].label
        labels = labels[labels != "S00"]
        this_label = util.find_mode(labels, default = "S00")
        utterances.loc[indexes, "label"] = this_label
        start = i
    inaudible = utterances[utterances.text == "[INAUDIBLE]"].index
    utterances.loc[inaudible, "label"] = "S00"
    return utterances
