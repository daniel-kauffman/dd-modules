#!/usr/bin/python2.7
#
# Voice Classifier Library

from __future__ import division, print_function
import argparse
import os
import traceback

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common.db as db
import common.util as util
import voice.extfeat as extfeat
import voice.gmm as gmm


def main():
    args = set_options()
    if args.verbose:
        print("Options:", str(vars(args)))
    utterances = db.query_utterances(wl_dict = {"file_id": [args.file_id]})
    utterances = utterances.rename(columns = {"pid": "label"})
    state = utterances.iloc[0].state
    pids = (db.query_legislators(state, year = 2015) if args.all else
            db.query_legislators(state, year = 2015, file_id = args.file_id))
    print(pids)
    clf = VoiceClassifier(state, pids, diarized = False, verbose = args.verbose)
    probas = clf.classify(utterances, clf_names = args.clfs)
    print(probas.loc[probas.index.get_level_values(0).isin(pids)])
    summed = probas.groupby(level = "label").sum()
    rankings = util.rank_probas(summed)
    scores = util.score_rankings(rankings)
    print(scores.loc[scores.index.get_level_values(0).isin(pids)])


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "{0:.3f}".format(x))
    
    parser = argparse.ArgumentParser(description = 
    """Test speaker identification voice classifiers in isolation.""")
    parser.add_argument("file_id", help = 
    """the File ID of a test video""")
    parser.add_argument("-a", "--all", action = "store_true", help = 
    """use models for all state legislators, rather than only speakers,
       during classification""")
    parser.add_argument("-c", "--clfs", choices = ["gmm", "svm"], nargs = "+",
                        default = ["gmm", "svm"], help = 
    """the classifiers to use for speaker identification""")
    parser.add_argument("-v", "--verbose", action = "store_true", help = 
    """print additional information to the terminal as the program is 
       executing""")
    return parser.parse_args()


def save_probas(probas, pkl_path):
    """
    Pickle a DataFrame of speaker probabilities to disk, appending data to a
    DataFrame at the same location if one exists.
    
    Args:
        probas: A DataFrame of speaker probabilities per cluster label.
        pkl_path: A str representing the location to pickle data.
    """
    if os.path.exists(pkl_path):
        pickled = pd.read_pickle(pkl_path)
#        probas = pickled.append(probas, ignore_index = True)
        probas = pickled.append(probas)
#        probas = probas.drop_duplicates(subset = columns, keep = "last")
#        probas = probas.reset_index(drop = True)
    probas.to_pickle(pkl_path)
    



class VoiceClassifier(object):
    
    def __init__(self, state, pids, deltas = 0, diarized = True,
                 verbose = False):
        self.params = {"state": state, "pids": sorted(pids), "deltas": deltas,
                       "verbose": verbose}
        self.diarized = diarized
        self.feat_factory = extfeat.FeatureFactory(state, deltas)
        self.clusters = None
    
    def get_model_paths(self):
        """
        Establish a base method to be overriden by derivative classes with
        model paths.
        
        Returns:
            An empty list.
        """
        return []
    
    def classify(self, segments, clf_names = ["gmm", "svm"]):
        """
        Assign a probability for each speaker per diarized segment using one
        or more given classifiers.
        
        Args:
            segments: A DataFrame containing segment data.
        
        Returns:
            A DataFrame of speaker probabilities per cluster label per
            classifier used.
        """
        file_id = segments.iloc[0].file_id
        clf_dict = {"gmm": GMMVoiceClassifier, "svm": SVMVoiceClassifier}
        if len(set(clf_names) - set(clf_dict.keys())) > 0:
            raise ValueError("Invalid Classifier Name(s)")
        clfs = [clf_dict[name](**self.params) for name in clf_names]
        fmt_str = "Voice Classification [Step {0:d} of {1:d}] ..."
        print(fmt_str.format(1, len(clf_names) + 1))
        clusters = self.cluster_features(segments)
        combined = pd.DataFrame()
        for i, clf in enumerate(clfs):
            print(fmt_str.format(i + 2, len(clf_names) + 1))
            clf.clusters = clusters
            probas = clf.classify(segments)
            combined = combined.append(probas)
        return combined[sorted(combined.columns)].sort_index()
    
    def cluster_features(self, segments):
        """
        Concatenate utterance features by cluster label (diarization ID).
        
        Args:
            segments: A DataFrame containing segment data.
        
        Returns:
            A dict of str label keys to concatenated 2D NumPy utterance
            feature arrays.
        """
        file_id = segments.iloc[0].file_id
        if self.diarized:
            pkl_path = "/audio/temp/{0}.npy".format(file_id)
            if os.path.exists(pkl_path):
                features = np.load(pkl_path)
            else:
                deltas = self.params["deltas"]
                features = extfeat.get_feature_matrix(segments, deltas = deltas)
                np.save(pkl_path, features)
        else:
            features = np.array(self.feat_factory.load_features(file_id))
        segments = segments.reset_index(drop = True)
        if len(segments) != len(features):
            raise ValueError("DataFrame and Feature Matrix Incompatible")
        clusters = {}
        for label in segments.label.unique():
            indexes = segments[segments.label == label].index.values - 1
            clusters[label] = np.concatenate(features[indexes])
        return clusters
    
    def format_probas(self, probas, file_id):
        """
        Conform a DataFrame of probabilities to have a specific MultiIndex.
        
        Args:
            probas: A DataFrame of speaker probabilities per cluster label.
            file_id: A str representing a File ID.
        
        Returns:
            A DataFrame indexed by cluster label, file ID, and classifier name.
        """
        probas.index.name = "label"
        probas["file_id"] = file_id
        probas["clf"] = str(self)
        return probas.set_index(["file_id", "clf"], append = True)
    
#    def join_by_label(self, segments, probas):
#        """
#        Perform a join on two DataFrames by cluster label.
#        
#        Args:
#            segments: A DataFrame containing segment data.
#            probas: A DataFrame with speaker probabilities for each segment.
#        
#        Returns:
#            A DataFrame of probabilities indexed by segment number.
#        """
#        joined = segments[["label"]].join(probas, on = "label")
#        return joined.drop("label", axis = 1)




class GMMVoiceClassifier(VoiceClassifier):
    """
    Perform voice recognition using Gaussian Mixture Models (GMMs).
    """
    
    def __init__(self, state, pids, deltas = 0, n_mix = 64, diarized = True,
                 verbose = False):
        super(GMMVoiceClassifier, self).__init__(state, pids, deltas,
                                                 diarized = diarized)
        self.gmm_factory = gmm.GMMFactory(state, n_mix, deltas)
        self.ubm = self.gmm_factory.load_gmm(verbose = verbose)
        self.models = {pid: self.gmm_factory.load_gmm(pid = pid,
                                                      verbose = verbose)
                       for pid in self.params["pids"]}
    
    def __call__(self, pid):
        try:
            if self.clusters is None:
                raise ValueError("Video Features Not Loaded")
            if not self.models[pid]:
                ratios = {label: None for label in self.clusters.keys()}
            else:
                ratios = {label: gmm.get_likelihood_ratio(self.models[pid],
                                                          self.ubm, features)
                          for label, features in self.clusters.items()}
            return pd.Series(ratios).rename(pid)
        except:
            print("PID:", pid)
            traceback.print_exc()
    
    def __repr__(self):
        return "vc_gmm_m{0}_d{1}".format(self.gmm_factory.n_mix,
                                         self.gmm_factory.deltas)
    
    def get_model_paths(self):
        """
        Collect a list of file paths for each model used by this classifier.
        
        Returns:
            A list of str, each representing a path to a GMM on disk.
        """
        return [self.gmm_factory.get_gmm_path(pid = pid)
                for pid, model in self.models.items() if model is not None]
    
    def classify(self, segments):
        """
        Assign a probability for each speaker per diarized segment.
        
        Args:
            segments: A DataFrame containing segment data.
        
        Returns:
            A DataFrame of speaker probabilities per cluster label.
        """
        if self.clusters is None:
            self.clusters = self.cluster_features(segments)
        ratios = pd.DataFrame(util.map_processes(self, self.models.keys()),
                              index = self.models.keys(),
                              columns = self.clusters.keys())
        ratios = ratios.T.sort_index()
        probas = ratios.apply(util.normalize_series, axis = 1)
        return self.format_probas(probas, segments.iloc[0].file_id)
    
    def __filter_segments(self, segments, ratios):
        """
        Remove segments not associated with a legislator.
        
        Args:
            segments: A DataFrame containing segment data.
            ratios: A DataFrame containing speaker likelihood ratios per
                    segment.
        
        Returns:
            A DataFrame with non-legislator segments removed.
        """
        pass




class SVMVoiceClassifier(VoiceClassifier):
    """
    Perform voice recognition using Support Vector Machines (SVMs).
    """
    
    def __init__(self, state, pids, deltas = 0, diarized = True):
        super(SVMVoiceClassifier, self).__init__(state, pids, deltas,
                                                 diarized = diarized)
        self.clf = self.__create_classifier()
    
    def __repr__(self):
        return "vc_svm"
    
    def classify(self, segments):
        """
        Assign a probability for each speaker per diarized segment.
        
        Args:
            segments: A DataFrame containing segment data.
        
        Returns:
            A DataFrame of speaker probabilities per cluster label.
        """
        if self.clusters is None:
            self.clusters = self.cluster_features(segments)
        self.clf.fit(*self.__load_training_data())
        series_list = []
        for label, features in self.clusters.items():
            averaged = self.__average_features(features)
            probas = self.clf.predict_proba(averaged)[0]
            series = pd.Series(probas, index = self.clf.classes_)
            series_list.append(series.rename(label))
        probas = pd.DataFrame(series_list)
        return self.format_probas(probas, segments.iloc[0].file_id)
    
    def __create_classifier(self, search = False):
        """
        Instantiate a SVM classifier using default or learned parameters.
        
        Args:
            search: A bool indicating whether to use cross validation to find
                    the best parameters for this classifier on a particular
                    feature set.
        
        Returns:
            A SVM classifier.
        """
#        self.scaler = RobustScaler(with_centering = scale,
#                                   with_scaling = scale)
#        self.scaler = StandardScaler(with_mean = scale, with_std = scale)
#        scaled = self.scaler.fit_transform(features)
        if not search:
            return SVC(C = 3, gamma = 0.005, probability = True)
        params = {"C": scipy.stats.expon(scale = 100),
                  "gamma": scipy.stats.expon(scale = 0.1)}
        return RandomizedSearchCV(SVC(probability = True), params,
                                  scoring = "precision_micro", n_iter = 20,
                                  n_jobs = 4, cv = 5, verbose = 1)
    
    def __load_training_data(self):
        """
        Load extracted features from disk to train this classifier.
        
        Returns:
            A 2-tuple containing a 2D array of features and a 1D array of
            the corresponding labels (Person IDs).
        """
        pids = self.params["pids"]
#        pid_feat = {pid: features
#                    for pair in util.map_processes(self.feat_factory, pids)
#                    for pid, features in pair.items() if len(features) > 0}
        load_features = self.feat_factory.load_features
        pid_feat = {pid: load_features(pid) for pid in pids}
        xs = []
        ys = []
        for pid, features in pid_feat.items():
            averaged = self.__average_features(features)
#            averaged = np.array([np.mean(features, axis = 0)])
            xs.append(averaged)
            ys.append([pid] * len(averaged))
        return (np.concatenate(xs), np.concatenate(ys))
    
    def __average_features(self, features, step = 200):
        """
        Calculate the mean for each set of features spanning a specific window.
        
        Args:
            features: A 2D NumPy array of features for one speaker.
            step: The number of features to increment by when forming windows
                  over which to average. The window size is twice this value.
        
        Returns:
            A 2D NumPy array of averaged features.
        """
        averaged = []
        for i in range(0, len(features), step):
            if i > 0 and i + step > len(features):
                break
            averaged.append(np.mean(features[i:i + step * 2, :], axis = 0))
        return np.array(averaged)


if __name__ == "__main__":
    main()
