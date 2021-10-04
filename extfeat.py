#!/usr/bin/python2.7
#
# Feature Extraction Library

from __future__ import division, print_function
import argparse
import os
import random
import traceback
import warnings

import features # https://github.com/jameslyons/python_speech_features
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav

import common.db as db
import common.util as util


def main():
    args = set_options()
    if args.verbose:
        print("Options:", str(vars(args)))
    hid_list_path = util.get_dir("pid") + "hids.txt"
    if args.extpid:
        with open(hid_list_path, "w"):
            pass
        for state in args.states:
            util.clean(util.get_dir("pid", state = state), extensions = ["npy"])
            util.clean(util.get_dir("vid", state = state), extensions = ["npy"])
            pids = db.query_legislators(state, verbose = args.verbose)
            feat_factory = FeatureFactory(state)
            util.map_processes(feat_factory, pids)
    if args.extvid:
        with open(hid_list_path, "r") as fp:
            hids = [int(line.strip()) for line in fp.readlines()]
        for state in args.states:
            n_test = min(utterances.file_id.nunique(), args.extvid)
            utterances = db.query_utterances(state, bl_dict = {"hid": hids})
            file_ids = random.sample(utterances.file_id.unique(), n_test)
            feat_factory = FeatureFactory(state)
            util.map_processes(feat_factory, file_ids)
            util.clean(util.get_dir("temp"), extensions = ["mp4", "wav"])


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    np.seterr(all = "ignore")   # override FloatingPointError
    states = db.query_states()
    
    parser = argparse.ArgumentParser(description =
    """Create a corpus of extracted audio features and speaker models.""")
    parser.add_argument("-p", "--extpid", action = "store_true", help =
    """extract audio features for training by speaker""")
    parser.add_argument("-s", "--states", nargs = "+", default = states,
                        choices = states, help =
    """the U.S. state legislature(s) from which to extract audio features""")
    parser.add_argument("-t", "--extvid", type = int, help =
    """the number of videos from which to extract audio features for testing""")
    parser.add_argument("-v", "--verbose", action = "store_true", help =
    """print additional information to the terminal as the program is
       executing""")
    return parser.parse_args()


def get_feature_matrix(utterances, concat = False, deltas = 2, verbose = False):
    """
    Create a matrix of audio features extracted from utterances.
    
    Args:
        utterances: A DataFrame containing utterance data.
        concat: A bool indicating whether to create a single matrix or an array
                of matrices in which each matrix represents a single utterance.
        deltas: An int indicating the level of delta-coefficients to use from
                the extracted audio features.
    
    Returns:
        A 2D NumPy array of concatenated features from one speaker or a
        3D NumPy array of utterance features from a video.
    """
    feat_list = []
    for i, row in utterances.iterrows():
        cut_path = cut_audio(row.file_id, row.start, row.end, verbose = verbose)
        if cut_path:
            features = extract_features(cut_path, deltas = deltas)
            feat_list.append(features)
            if os.path.exists(cut_path):
                os.remove(cut_path)
    return np.concatenate(feat_list) if concat else np.array(feat_list)


def cut_audio(file_id, start, end, verbose = False):
    """
    Cut a .wav file at given start and end positions.
    
    Args:
        file_id: A str representing a File ID.
        start: An int indicating a starting cut position in milliseconds.
        end: An int indicating an ending cut position in milliseconds.
    
    Returns:
        A str containing the path of the newly cut file.
    """
    if end - start > 0:
        temp = util.get_dir("temp")
        wav_path = temp + "{0}.wav".format(file_id)
        if not os.path.exists(wav_path):
            vid_path = util.download_video_from_s3(file_id, verbose = verbose)
            if not vid_path:
                return
            util.convert_to_wav(vid_path, verbose = verbose)
        start_code = util.get_time_code(start)
        length_code = util.get_time_code(end - start)
        cut_path = temp + "{0}-{1:07d}.wav".format(file_id, start)
        if verbose:
            print(cut_path)
        command = ["ffmpeg", "-i", wav_path, "-ss", start_code, 
                   "-t", length_code, "-acodec", "copy", "-y", cut_path]
        util.run_silently(command)
        if os.path.exists(cut_path):
            return cut_path


def extract_features(wav_path, deltas = 2): 
    """
    Extract MFCC audio features from an audio file.
    
    Args:
        wav_path: A str containing the path to the audio file.
        deltas: An int indicating the level of delta-coefficients to use from
                the extracted audio features.
                
                0: Use no delta-coefficients.
                1: Use differential delta-coefficients.
                2: Use differential and acceleration delta-coefficients.
    
    Returns:
        A 2D NumPy array with each row representing the audio features of a
        particular time step.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signal = wav.read(wav_path)[1]
    
    frames = features.mfcc(signal, numcep = 26)
    frames -= frames.mean(axis = 0)     # Cepstral Mean Subtraction
    if deltas:
        differential = get_deltas(frames)
        frames =  np.concatenate((frames, differential), axis = 1)
        if deltas > 1:
            acceleration = get_deltas(differential)
            frames =  np.concatenate((frames, acceleration), axis = 1)
    return frames


def get_deltas(frames, span = 2):
    """
    Calculate the delta-coefficients of a feature matrix.
    
    Args:
        frames: A 2D NumPy array of extracted audio features.
        span: An int specifying the number of adjacent frames over which
              changes are to be measured.
    
    Returns:
        A 2D NumPy array of delta-coefficients.
    """
    deltas = np.zeros(np.shape(frames))
    min_i, max_i = 0, len(frames) - 1
    divisor = 2.0 * sum([n ** 2 for n in range(1, span + 1)])
    for i, frame in enumerate(frames):
        deltas[i] = np.sum([n * (frames[min(max_i, i + n)] -
                                 frames[max(min_i, i - n)])
                            for n in range(1, span + 1)], axis = 0)
    return deltas / divisor


def cut_utterances(utterances, cut_len = 1):
    """
    Remove initial and trailing seconds from utterances.
    
    Args:
        utterances: A DataFrame containing utterance data.
        cut_len: An int specifying the number of seconds to cut at both ends of
                 every utterance.
    
    Returns:
        A DataFrame of utterances with start and end columns updated.
    """
    utterances = utterances.copy()
    utterances.start += cut_len * 1000
    utterances.end -= cut_len * 1000
    return utterances


def select_utterances(utterances, max_sec, greedy = True):
    """
    Select a subset of utterances such that their total duration is less than
    a given threshold.
    
    Args:
        utterances: A DataFrame containing utterance data.
        max_sec: An int specifying the maximum value for the total duration of
                 the utterances selected.
        greedy: A bool indicating whether to use as few utterances as possible
                to reach max_sec.
    
    Returns:
        A DataFrame containing a subset of utterance data.
    """
    utterances = sort_utterances(utterances, ascending = not greedy)
    duration = 0
    selected = []
    for _, row in utterances.iterrows():
        selected.append(row)
        duration += row.duration / 1000
        if duration > max_sec:
            break
    return pd.DataFrame(selected)


def sort_utterances(utterances, ascending = False):
    """
    Sort a DataFrame of utterance data by each File ID's total duration.
    
    Args:
        utterances: A DataFrame containing utterance data.
        ascending: A bool indicating whether to sort in ascending (File IDs
                   with lowest total durations first) or descending order.
    
    Returns:
        A DataFrame containing sorted utterance data.
    """
    utterances = utterances.copy()
    sums = utterances.groupby("file_id")["duration"].sum().rename("sum")
    utterances = utterances.join(sums, on = "file_id")
    utterances = utterances.sort_values(["sum", "duration"],
                                        ascending = ascending)
    utterances = utterances.drop("sum", axis = 1)
    return utterances




class FeatureFactory(object):
    """
    Extract and archive audio features by speaker or by video.
    """
    
    def __init__(self, state, deltas = 2):
        self.state = state
        self.deltas = deltas
        self.params = {"pid": None, "file_id": None}
    
    def __call__(self, param):
        try:
            self.__set_params(param)
            self.__save_features()
        except:
            print("FeatureFactory Failure:", param)
            traceback.print_exc()
    
    def __set_params(self, param):
        """
        Set the PID and File ID values of the object's params dict.
        
        Args:
            param: An int or str representing a PID or File ID, respectively.
        """
        if type(param) == int or type(param) == np.int64:
            self.params["pid"] = param
            self.params["file_id"] = None
        elif type(param) == str or type(param) == unicode:
            self.params["pid"] = None
            self.params["file_id"] = param
        else:
            TypeError("Invalid Parameter Type:", type(param))
    
    def load_features(self, param):
        """
        Load speaker or video audio features pickled to disk.
        
        Args:
            param: An int or str representing a PID or File ID, respectively.
        
        Returns:
            A 2D NumPy array or an array of 2D NumPy arrays, with the number of
            delta-coefficients loaded based on the object's deltas attribute.
        """
        self.__set_params(param)
        feat_path = self.__get_feat_path()
        if not os.path.exists(feat_path):
            self.__save_features(param)
        if not os.path.exists(feat_path):
            raise IOError("Features Not Found:", param)
        features = np.load(feat_path)
        if self.deltas < 2:
            numcep = 26
            n_cols = numcep * (self.deltas + 1)
            if len(features.shape) == 2:
                features = features[:, :n_cols]
            else:
                features = np.array([utt_feat[:, :n_cols]
                                     for utt_feat in features])
        return features
    
    def __save_features(self, max_sec = 600, min_len = 2):
        """
        Extract audio features for a specific speaker across multiple videos
        and pickle them to disk.
        
        Args:
            max_sec: An int specifying the maximum value for the total duration
                     of the utterances selected for extraction.
            min_len: An int specifying the minimum duration of any utterance
                     selected for extraction.
        
        Returns:
            A 2D NumPy array of extracted audio features.
        
        Raises:
            ValueError: There is not enough utterance data for the speaker.
        """
        if self.params["pid"]:
            pid = self.params["pid"]
            cut_len = 1
            min_len = min_len + 2 * cut_len
            utterances = db.query_utterances(self.state, min_len = min_len,
                                             wl_dict = {"pid": [pid]})
            if utterances.duration.sum() / 1000 < 10:
                raise ValueError("Insufficient Voice Data for PID: " + str(pid))
            utterances = cut_utterances(utterances, cut_len = cut_len)
            utterances = select_utterances(utterances, max_sec)
            with open(util.get_dir("pid") + "hids.txt", "a") as fp:
                hids = utterances.hid.unique().tolist()
                fp.write("\n".join([str(hid) for hid in hids]) + "\n")
        elif self.params["file_id"]:
            wl_dict = {"file_id": [self.params["file_id"]]}
            utterances = db.query_utterances(self.state, wl_dict = wl_dict)
        else:
            raise ValueError("Person ID or File ID Required")
        feat_path = self.__get_feat_path()
        features = get_feature_matrix(utterances,
                                      concat = bool(self.params["pid"]))
        np.save(feat_path, features)
    
    def __get_feat_path(self):
        """
        Create a path for pickled extracted audio features, using different
        directories depending on whether the features belong to a single
        speaker or come from a single video.
        
        Returns:
            A str containing the location of the pickled features.
        
        Raises:
            ValueError: Neither a PID nor a File ID is found.
        """
        if self.params["pid"]:
            feat_path = (util.get_dir("pid", state = self.state)
                         + "{0:09d}.npy".format(self.params["pid"]))
        elif self.params["file_id"]:
            feat_path = (util.get_dir("vid", state = self.state)
                         + "{0}.npy".format(self.params["file_id"]))
        else:
            raise ValueError("Person ID or File ID Required")
        return feat_path


if __name__ == "__main__":
    main()
