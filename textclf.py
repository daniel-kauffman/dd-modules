#!/usr/bin/python2.7
#
# Utterance Text Classification Library

from __future__ import division, print_function
import argparse
import cPickle
import itertools
import os
import pprint

import nltk
from nltk import tag
from sklearn.metrics import precision_recall_fscore_support as prfs

import pandas as pd

import common.metrics as metrics
import common.query as query
import common.util as util


def main():
    args = set_options()
    if args.verbose:
        print("Options:", str(vars(args)))
    utterances = query.query_finalized_utterances(file_ids = [args.file_id])
    utterances = utterances.rename(columns = {"pid": "label"})
    state = utterances.iloc[0].state
    tic = TextIntroClassifier(state, verbose = args.verbose)
    utterances = tic.classify(utterances)
    print(utterances)
    print(tic.score(utterances))


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "{0:.5f}".format(x))
    
    parser = argparse.ArgumentParser(description = 
    """Test speaker identification voice classifiers in isolation.""")
    parser.add_argument("file_id", help = 
    """the File ID of a test video""")
    parser.add_argument("-v", "--verbose", action = "store_true", help = 
    """print additional information to the terminal as the program is 
       executing""")
    return parser.parse_args()


def preprocess(utterances):
    """
    Tokenize and perform named entity recognition on the given string.
    
    Args:
        text: A str containing one or more sentences.
    
    Returns:
        A list of list of str, with each inner list containing processed tokens
        for a single sentence.
    """
    utterances = utterances.copy()
    text = "\0".join(utterances.text.values)
    ner_path = ("jar/stanford-ner-2016-10-31/classifiers/" +
                "english.all.3class.distsim.crf.ser.gz")
    jar_path = "jar/stanford-ner-2016-10-31/stanford-ner-3.7.0.jar"
    ner = tag.StanfordNERTagger(ner_path, jar_path)
    tokenized = [nltk.wordpunct_tokenize(utterance)
                 for utterance in text.split("\0")]
    tagged = ner.tag_sents(tokenized)
    utt_list = []
    for i, utterance in enumerate(tagged):
        utt_tokens = []
        groups = itertools.groupby(utterance, key = lambda pair: pair[1])
        for ne_tag, group in groups:
            if ne_tag != "O":   # IOB "Outside" tag
                utt_tokens.append([str(label) for _, label in group][0])
            else:
                for token, _ in group:
                    try:
                        token = str(token).strip().lower()
                        if len(token) > 0 and token.isalpha():
                            utt_tokens.append(token)
                    except:
                        pass
        utterances.set_value(utterances.iloc[i].name, "text",
                             " ".join(utt_tokens))
    return utterances




class TextClassifier(object):
    
    def __init__(self, state, verbose = False):
        self.state = state
        self.verbose = verbose
        self.most_common = 300
        fmt_str = ("text/pkl/{0}".format(self.__class__.__name__.upper()) +
                   "_{0}_" + "{0}.pkl".format(state.upper()))
        names = ["ngram-list", "ngram-cfd", "pos-utt", "neg-utt"]
        self.pkl_paths = {name: fmt_str.format(name.upper()) for name in names}
        self.ngrams = self.__load_ngrams()
    
    def classify(self, utterances):
        """
        Join a column of bool to the given utterance DataFrame indicating
        whether each utterance is positive.
        
        Args:
            utterances: A DataFrame containing utterance data.
        
        Returns:
            A DataFrame with the same schema as the input but with an
            additional positive match column.
        """
        utterances = preprocess(utterances)
        is_pos = utterances.text.str.contains("|".join(self.ngrams))
        return utterances.join(is_pos.rename("pos"))
    
    def score(self, utterances):
        """
        Calculate the precision, recall, and F0.5 score of this classifier's
        predictions.
        
        Args:
            utterances: A DataFrame containing utterance data that has
                        undergone classification.
        
        Returns:
            A Series containing precision, recall, and f_score attributes.
        """
        leg_pids = set(query.query_legislators(self.state, year = 2015))
        leg_pids |= set(query.query_legislators(self.state, year = 2017))
        is_not_leg = lambda row: row.name not in leg_pids
        actual = utterances.groupby("label").apply(is_not_leg).rename("actual")
        pred = utterances.groupby("label").pos.any().rename("pred")
        p, r, f, _ = prfs(actual, pred, beta = 0.5, pos_label = True,
                          average = "binary")
        if self.verbose:
            errors = {"fp": {}, "fn": {}}
            for _, row in utterances.iterrows():
                if row.pos and row.label in leg_pids:
                    errors["fp"].setdefault(row.label, []).append(row.text)
            for label, row in pd.DataFrame([actual, pred]).T.iterrows():
                if not row.pred and label not in leg_pids:
                    text = utterances[utterances.label == label].text
                    errors["fn"][label] = text.values.tolist()
            pprint.pprint(errors)
        return pd.Series({"precision": p, "recall": r, "f_score": f},
                         index = ["precision", "recall", "f_score"])
    
    def query_utterances(self, query):
        """
        
        
        Args:
            query: A str containing a MySQL query.
        
        Returns:
            A DataFrame with the following schema:
                pid      int64
                name    object
                text    object
        """
        records = util.query_database(query, index_col = "uid",
                                      verbose = self.verbose)
        records.text = records.text.astype(str)
        if set(records.columns.values) != set(["pid", "name", "text"]):
            raise ValueError("Invalid Schema")
        return records
    
    def __extract_ngrams(self):
        """
        Create ngrams from a training set that maximizes the F0.5 score for
        identifying this classifier's target utterance characteristics.
        
        Returns:
            A NumPy array of str, each an ngram from a training set.
        """
        pos_utts = self.__load_utterances(30000)
        neg_utts = self.__load_utterances(20000, negate = True)
        training_uids = pos_utts.sample(10000).index
        pos_cfd = self.__load_ngram_cfd(pos_utts.loc[training_uids])
        pos_utts = pos_utts[~pos_utts.index.isin(training_uids)]
        print("\nCounting Occurrences ...")
        counts = self.__count_ngrams(pos_utts, neg_utts, pos_cfd)
        print("\nCollecting UIDs ...")
        ngram_uids = self.__get_ngram_uids(pos_utts, pos_cfd)
        return self.__select_ngrams(counts, ngram_uids).index.values
    
    def __count_ngrams(self, pos_utts, neg_utts, pos_cfd):
        """
        Count the number of occurrences of the most common ngrams extracted
        from a training set of positive utterances in a testing set of both
        positive and negative utterances.
        
        Args:
            pos_utts: A DataFrame of positive utterances.
            neg_utts: A DataFrame of negative utterances.
            pos_cfd: A ConditionalFreqDist with ngram sizes as conditions and
                       FreqDists of ngrams extracted from a training set of
                       positive utterances.
        
        Returns:
            A DataFrame with the following schema:
                pos        float64
                neg        float64
                precision    float64
        """
        series_list = []
        for fd in pos_cfd.values():
            for ngram, _ in fd.most_common(self.most_common):
                pos_contains = pos_utts.text.str.contains(ngram)
                neg_contains = neg_utts.text.str.contains(ngram)
                pos_counts = pos_contains.value_counts().get(True, 0)
                neg_counts = neg_contains.value_counts().get(True, 0)
                precision = pos_counts / float(pos_counts + neg_counts)
                series = pd.Series({"pos": pos_counts, "neg": neg_counts,
                                    "precision": precision})
                series_list.append(series.rename(ngram))
        columns = ["pos", "neg", "precision"]
        counts = pd.DataFrame(series_list, columns = columns)
        counts = counts.sort_values("precision", ascending = False)
        counts.pos = counts.pos.astype(int)
        counts.neg = counts.neg.astype(int)
        return counts
    
    def __get_ngram_uids(self, pos_utts, pos_cfd):
        """
        Collect all unique Utterance IDs (UIDs) for each ngram in which that
        ngram occurs.
        
        Args:
            pos_utts: A DataFrame of positive utterances.
            pos_cfd: A ConditionalFreqDist with ngram sizes as conditions and
                       FreqDists of ngrams extracted from a training set of
                       positive utterances.
        
        Returns:
            A dict with ngrams as keys and uid sets as values.
        """
        ngram_uids = {}
        for fd in pos_cfd.values():
            for ngram, _ in fd.most_common(self.most_common):
                pos_contains = pos_utts.text.str.contains(ngram)
                if pos_contains.any():
                    matches = pos_contains[pos_contains == True]
                    ngram_uids[ngram] = set(matches.index.values)
        if self.verbose:
            print("UIDs:", len(set(itertools.chain(*ngram_uids.values()))))
        return ngram_uids
    
    def __select_ngrams(self, counts, ngram_uids):
        """
        Select a subset of ngrams that maximize the classifier's F0.5 score.
        
        Args:
            counts:
            ngram_uids:
        
        Returns:
            A DataFrame with the following schema:
                pos        float64
                neg        float64
                precision    float64
        """
        dict_list = []
        model_set = set()
        acc_pos = 0
        acc_neg = 0
        total_uids = len(set(itertools.chain(*ngram_uids.values())))
        for i, row in counts.iterrows():
            ngram = row.name
            model_set |= ngram_uids.get(ngram, set())
            acc_pos += row.pos
            acc_neg += row.neg
            precision = acc_pos / float(acc_pos + acc_neg)
            recall = len(model_set) / float(total_uids)
            f_score = metrics.calculate_fscore(precision, recall, beta = 0.5)
            dict_list.append({"acc_pre": precision, "acc_rec": recall,
                              "f_score": f_score})
        columns = ["acc_pre", "acc_rec", "f_score"]
        f_scores = counts.join(pd.DataFrame(dict_list, index = counts.index,
                                            columns = columns))
        if self.verbose:
            print(f_scores, "\n")
            print(f_scores.loc[f_scores.f_score.idxmax()], "\n")
        return counts.loc[:f_scores.f_score.idxmax()]
    
    def __load_ngrams(self):
        """
        Create, if necessary, an array of ngrams for text classification.
        
        Returns:
            A NumPy array of str, each an ngram from a training set.
        """
        pkl_path = self.pkl_paths["ngram-list"]
        if os.path.exists(pkl_path):
            with open(pkl_path, "r") as fp:
                ngrams = cPickle.load(fp)
        else:
            ngrams = self.__extract_ngrams()
            with open(pkl_path, "w") as fp:
                cPickle.dump(ngrams, fp)
        return ngrams
    
    def __load_utterances(self, count, negate = False):
        """
        Create, if necessary, a DataFrame of normalized, NER-tagged utterances.
        
        Args:
            count: The number of utterances to query.
        
        Returns:
            A DataFrame with the following schema:
                pid      int64
                name    object
                text    object
        """
        pkl_path = (self.pkl_paths["pos-utt"]
                    if not negate else self.pkl_paths["neg-utt"])
        if os.path.exists(pkl_path):
            return pd.read_pickle(pkl_path)
        utterances = preprocess(self.query_utterances(self.state, count,
                                                      negate = negate))
        utterances.to_pickle(pkl_path)
        return utterances
    
    def __load_ngram_cfd(self, utterances, ngram_sizes = range(3, 7)):
        """
        Create, if necessary, a ConditionalFreqDist of ngrams extracted from a
        training set of utterances.
        
        Args:
            utterances: A DataFrame containing utterance data.
            ngram_sizes: A list of int, each an ngram size for a CFD condition.
        
        Returns:
            A ConditionalFreqDist with ngram sizes as conditions and FreqDists
            of ngrams of the condition's size.
        """
        pkl_path = self.pkl_paths["ngram-cfd"]
        if os.path.exists(pkl_path):
            with open(pkl_path, "r") as fp:
                return cPickle.load(fp)
        cfd = (nltk.ConditionalFreqDist((n, " ".join(ngram))
               for n in ngram_sizes for utterance in utterances.text.values
               for ngram in nltk.ngrams(utterance.split(), n)))
        with open(pkl_path, "w") as fp:
            cPickle.dump(cfd, fp)
        return cfd


class TextIntroClassifier(TextClassifier):
    
    def __init__(self, state, verbose = False):
        super(TextIntroClassifier, self).__init__(state, verbose = verbose)
    
    def query_utterances(self, state, count, negate = False, max_len = 1000):
        """
        Query a specified number of utterances considered to be
        self-introductions (those in which the speaker states their full name).
        
        Args:
            state: A two-character str representing a U.S. state abbreviation.
            count: An int specifying the number of utterances to retrieve.
            negate: A bool indicating whether to retrieve non-introductory
                    utterances (those not containing the speaker's full name).
        
        Returns:
            A DataFrame with the following schema:
                pid      int64
                name    object
                text    object
        """
        query = """
            SELECT u.uid, u.pid, CONCAT(p.first, " ", p.last) AS name,
                   CONVERT(u.text USING ASCII) AS text
            FROM Person p, currentUtterance u
            WHERE p.pid = u.pid
              AND u.state = "{0}"
              AND CHAR_LENGTH(u.text) < {1:d}
              AND u.text {2}LIKE BINARY CONCAT("%", p.first, " ", p.last, "%")
            ORDER BY RAND()
            LIMIT {3:d};
                """.format(state, max_len, "NOT " if negate else "", count)
        return super(TextIntroClassifier, self).query_utterances(query)




class TextBillAuthorClassifier(TextClassifier):
    
    def __init__(self, state, verbose = False):
        super(TextBillAuthorClassifier, self).__init__(state, verbose = verbose)


if __name__ == "__main__":
    main()
