#!/usr/bin/python2.7
#
# GMM Speaker Model Program

from __future__ import division, print_function
import argparse
import glob
import math
import os
import pickle
import traceback

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

import common.db as db
import common.util as util
import voice.extfeat as extfeat


def main():
    args = set_options()
    if args.verbose:
        print("Options:", str(vars(args)))
    for state in args.states:
        n_mix, deltas = args.model if args.model else (64, 0)
        is_power_of = lambda x: math.log(n_mix, x) % 1 == 0
        if not is_power_of(2) or deltas not in range(3):
            raise ValueError("Invalid GMM Parameters")
        dir_path = util.get_dir("pid", state = state)
        pattern = "P*_M{0:04d}_D{1:d}.gmm".format(n_mix, deltas)
        gmm_paths = glob.glob(util.get_dir("gmm", state = state) + pattern)
        for path in gmm_paths:
            os.remove(path)
        pids = [int(util.get_root_name(path))
                for path in glob.glob(dir_path + "*.npy")]
        gmm_factory = GMMFactory(state, n_mix, deltas)
        gmm_factory.load_gmm()
        util.map_processes(gmm_factory, pids)


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    np.seterr(all = "ignore")   # override FloatingPointError
    states = db.query_states()
    
    parser = argparse.ArgumentParser(description =
    """Create a corpus of extracted audio features and speaker models.""")
    parser.add_argument("-m", "--model", nargs = 2, type = int,
                        metavar = ("MIXTURES", "DELTAS"), help =
    """create speaker models with given mixtures and deltas""")
    parser.add_argument("-s", "--states", nargs = "+", choices = states,
                        default = states, help =
    """the U.S. state legislature for which to extract audio features
       (perform on all if not specified)""")
    parser.add_argument("-v", "--verbose", action = "store_true", help =
    """print additional information to the terminal as the program is
       executing""")
    return parser.parse_args()


def get_likelihood_ratio(hyp, ubm, features):
    """
    Calculate the likelihood ratio between two GMMs.
    
    Args:
        hyp: A GMM used as the hypothetical speaker model.
        ubm: A GMM used as the Universal Background Model.
        features: A list of NumPy arrays, one per segment.
    
    Returns: A float representing the likelihood ratio that the hypothesis more
             closely aligns with the features than the background model.
    """
    return np.sum(hyp.score(features) - ubm.score(features))


def evaluate_ratios(ratios, threshold = -1.0):
    """
    Evaluate the accuracy of likelihood ratios.
    
    Args:
        ratios: A DataFrame of likelihood ratios for each speaker by segment.
        threshold: A float which, if all ratios for a segment are less than,
                   indicates the segment is for a non-legislator.
    
    Returns:
        A Series containing TP, TN, FP, and FN metrics regarding whether
        non-legislator segments were correctly identified.
    """
    ratios = ratios[ratios.actual.notnull()]
    pids = [pid for pid in ratios if type(pid) == int]
    tp = len(ratios[(ratios.actual.isin(pids)) &
                    (ratios[pids] > threshold).any(axis = 1)])
    tn = len(ratios[(~ratios.actual.isin(pids)) &
                    (ratios[pids] <= threshold).all(axis = 1)])
    fp = len(ratios[(~ratios.actual.isin(pids)) &
                    (ratios[pids] > threshold).any(axis = 1)])
    fn = len(ratios[(ratios.actual.isin(pids)) &
                    (ratios[pids] <= threshold).all(axis = 1)])
    precision = 0 if tp + fp == 0 else tp / float(tp + fp)
    recall = 0 if tp + fn == 0 else tp / float(tp + fn)
    f2_score = util.calculate_fscore(precision, recall, beta = 2)
    index = ["tp", "tn", "fp", "fn", "f2_score"]
    return pd.Series({"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                      "f2_score": f2_score}, index = index)




class GMMFactory:
    """
    Create and load Gaussian Mixture Models (GMMs) by Person ID and GMM-based
    Universal Background Models (UBMs) by state.
    """
    
    def __init__(self, state, n_mix, deltas):
        self.state = state
        self.n_mix = n_mix
        self.deltas = deltas
        self.feat_factory = extfeat.FeatureFactory(self.state, self.deltas)
    
    def __call__(self, pid):
        try:
            self.load_gmm(pid = pid)
        except:
            print("GMMFactory Failure:", pid)
            traceback.print_exc()
    
    def get_gmm_path(self, pid = None):
        """
        Create a path for saving and loading a GMM.
        
        Args:
            pid: An int indicating the Person ID of the GMM. If None, create a
                 path for a UBM.
        
        Returns:
            A str representing the path to a GMM.
        """
        prefix = ("UBM-" + self.state.upper()
                  if pid is None else "P{0:09d}".format(pid))
        fmt_str = prefix + "_M{0:04d}_D{1:d}.gmm"
        gmm_dir = util.get_dir("gmm", state = self.state)
        return gmm_dir + fmt_str.format(self.n_mix, self.deltas)
    
    def load_gmm(self, pid = None, verbose = False):
        """
        Load a GMM, including a UBM.
        
        Args:
            pid: An int indicating the Person ID of the GMM to load. If None,
                 load a UBM.
        
        Returns:
            A GMM corresponding to the given Person ID or UBM.
        """
        gmm_path = self.get_gmm_path(pid = pid)
        if os.path.exists(gmm_path):
            with open(gmm_path, "rb") as fp:
                return pickle.load(fp)
        try:
            return self.__save_gmm(pid = pid, verbose = verbose)
        except:
            traceback.print_exc()
    
    def __save_gmm(self, pid = None, verbose = False):
        """
        Create and save a GMM, including a UBM.
        
        Args:
            pid: An int indicating the Person ID of the GMM to create. If None,
                 create a UBM.
        
        Returns:
            A GMM corresponding to the given Person ID or UBM.
        """
        if pid:
            if verbose:
                print("Creating Speaker Model ({0}) ...".format(pid))
            features = self.feat_factory.load_features(pid)
        else:
            if verbose:
                print("Creating Universal Background Model (UBM) ...")
            utterances = self.get_public_utterances()
            file_ids = utterances.file_id.unique().tolist()
            vid_paths = util.map_processes(util.download_video_from_s3,
                                           file_ids)
            vid_paths = [path for path in vid_paths if path is not None]
            util.map_processes(util.convert_to_wav, vid_paths)
            features = extfeat.get_feature_matrix(utterances, concat = True,
                                                  deltas = self.deltas,
                                                  verbose = verbose)
        if features is not None:
            kwargs = {"n_components": self.n_mix, "covariance_type": "tied"}
            gmm = GaussianMixture(**kwargs).fit(features)
            gmm_path = self.get_gmm_path(pid = pid)
            with open(gmm_path, "wb") as fp:
                pickle.dump(gmm, fp)
            return gmm
    
    def get_public_utterances(self, utt_len = 10, max_sec = 3600,
                              verbose = False):
        """
        Query utterances spoken by non-legislators.
        
        Args:
            utt_len: An int indicating the number of seconds for each utterance;
                     if an utterance's length exceeds this value, truncate it.
            max_sec: An int indicating the total number of seconds of utterance
                     data to retrieve.
        
        Returns:
            A DataFrame of utterance data, each from a non-legislator.
        """
        bl_dict = {"pid": db.query_legislators(self.state)}
        utterances = db.query_utterances(self.state, bl_dict = bl_dict,
                                         min_len = utt_len)
        utterances.end = utterances.start + utt_len * 1000
        return utterances.groupby("pid").first().sample(n = max_sec // utt_len)


if __name__ == "__main__":
    main()
