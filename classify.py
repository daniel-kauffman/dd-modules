#!/usr/bin/python2.7
#
# Voice-Face-Text (VFT) Driver

from __future__ import division, print_function
import argparse
import os

import matplotlib as mpl
mpl.use("Agg")  # required before pandas import
import pandas as pd

import common.db as db
import common.util as util
import text.srt as srt
import voice.voiceclf as vc
import traceback


SUPPORTED_STATES = {"CA", "NY"}


def main():
    args = set_options()
    if args.verbose:
        print("Options:", str(vars(args)))
    if "." not in args.file:
        args.file += ".srt"
    classify(args.file, args.cids, args.pids, srt_text = None,
             text = args.text, face = args.face,
             voice = args.skl_voice,
             testing = True, verbose = args.verbose)


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "{0:.3f}".format(x))
    
    parser = argparse.ArgumentParser(description = 
    """Perform Voice-Face-Text (VFT) speaker identification.""")
    parser.add_argument("file", nargs = "?", metavar = "SRT file", help = 
    """the path of the SRT file to diarize""")
    parser.add_argument("-c", "--cids", nargs = "*", type = int,
                        default = [], help = 
    """the committees associated with the video (hearing)""")
    parser.add_argument("-f", "--face", action = "store_true", help = 
    """use face recognition""")
    parser.add_argument("-p", "--pids", nargs = "*", type = int,
                        default = [], help = 
    """additional known attending legislators, such as bill authors""")
    parser.add_argument("-s", "--skl-voice", action = "store_true", help = 
    """use scikit-learn classifiers for voice recognition""")
    parser.add_argument("-t", "--text", action = "store_true", help = 
    """use text classification to filter out non-legislator utterances""")
    parser.add_argument("-v", "--verbose", action = "count", help = 
    """print additional information to the terminal as the program is 
       executing""")
    return parser.parse_args()


def classify(file_name, cids, pids, srt_text, state = None, text = False,
             face = False, voice = False, testing = False, verbose = False,
             utterances = None):
    """
    Identify the speakers of a video by diarizing it into segments and using
    characteristics of text, face, and voice.
    
    Args:
        file_name: A str representing a File ID.
        cids: A list of int representing Commmittee IDs.
        pids: A list of int representing Person IDs.
        srt_text: A str containing automated text from a transcription service.
        state: A str containing the US state abbreviation for the video. 
        text: A bool indicating whether to use text classification.
        face: A bool indicating whether to use face classification.
        voice: A bool indicating whether to use voice classification.
        testing: A bool indicating whether to run in testing mode.
        utterances: If None then classify will perform it's own diarization.
            However, if not None then classify will not run diarization and will
            use the utterances given. This allows diarization to happen externally.

    Returns:
        A tuple containing a pair of DataFrames, one containing diarized
        segment data and the other containing speaker probability data.
    
    Raises:
        ValueError: No classification method was selected.
        ValueError: No Person IDs were given or could be found by Committee ID.
    """
    if not any([text, face, voice]):
        raise ValueError("No Classification Method Selected")
    file_id = util.get_root_name(file_name)
    if state is None:
        query = """
                SELECT state
                FROM Video
                WHERE fileId = "{0}";
                """.format(file_id)
        state = db.query_database(query, verbose = verbose).state[0]
    
    state = state.upper()
    if state not in SUPPORTED_STATES:
        fmt_str = "State '{}' not currently supported, no models ready yet!"
        raise Exception(fmt_str.format(state))
    
    if utterances is None:
        # diarize video if not done previously
        utterances = srt.load_utterances(file_id, state, srt_text = srt_text,
                                         verbose = verbose)
    probas = None
    update_probas = lambda new_probas: (new_probas if probas is None else
                                        probas.add(new_probas, fill_value = 0))
    if pids is None:
        pids = []
    pids = (db.query_pids(file_id, verbose = verbose) if testing else
            sorted(set(pids + db.query_pids_by_committees(cids))))
#    pids = db.query_legislators(state, year = 2017)
    if len(pids) == 0:
        raise ValueError("No Person IDs Found")
    if verbose:
        print("PIDs:", pids)
    if voice:
        clf = vc.VoiceClassifier(state, pids, verbose = verbose)
        probas = update_probas(clf.classify(utterances, clf_names = ["gmm"]))

    if face:
        print("\nFace Classifying...")
        face_probas = classify_by_face(utterances, state, pids, verbose = verbose)

        if face_probas is not None:
            probas = update_probas(face_probas)
    
    probas = probas.fillna(0)

    if probas.empty:
        raise ValueError("No Predictions Found")
    if utterances is not None and probas is not None:
        is_normalized = lambda df: round(df.apply(sum, axis = 1).max(), 1) == 1
        if not is_normalized(probas):
            probas = probas.apply(util.normalize_series, axis = 1,
                                  robust = False)
        try:
            srt_path = util.get_dir("srt") + file_id + ".srt"
            write_srt(utterances, probas, srt_path)
        except IOError:
            print("\nFailed to write SRT!")
            traceback.print_exc()
        # task_probe.py writes SRT when not testing
        return (utterances, probas)


def classify_by_face(segments, state, pids, verbose = False):
    """
    Identify speakers of diarized segments by face.
    
    Args:
        segments: A DataFrame containing segment data in each row.
        state: A two-character str representing a U.S. state abbreviation.
        pids: A list of int representing Person IDs.
    
    Returns:
        A DataFrame of speaker probabilities per diarization cluster label.
    """
    import face.faceclf as fc
    # model_path = "/dd-data/models/production_network_135_epochs.pkl.gz"
    clf = fc.FaceClassifier(None, use_state_models = True)
    if not clf.has_model_for_state(state):
        print("Face has no model ready for state '{}'".format(state))
        return None
    utt_dicts = segments.to_dict(orient = "records")
    # did_preds: {did: [(pid, conf), ...]}
    did_preds = clf.classify(utt_dicts, pids = pids, vid_state = state,
                             verbose = verbose)
    # convert to: {did: {pid: conf}, ...}
    did_preds_dict = dict() 
    for did, preds in did_preds.iteritems():
        did_preds_dict[did] = {pid : conf for pid, conf in preds}
    probas = pd.DataFrame(did_preds_dict).T
    probas = probas.fillna(0)
    probas.index.name = "label"
    return probas


def write_srt(segments, probas, srt_path, verbose = False):
    """
    Write segment data with speaker predictions to a file in the .srt format.
    
    Args:
        segments: A DataFrame containing segment data in each row.
        probas: A DataFrame with speaker probabilities for each segment.
        srt_path: A str representing the location of a .srt file.
    
    Returns:
        A tuple containing the .srt path and its text.
    """
    srt_text = make_srt(segments, probas)
    with open(srt_path, "w") as srt_file:
        srt_file.writelines(srt_text)
    if os.path.exists(srt_path):
        if verbose:
            print(srt_path, "written to disk")
        return (srt_path, srt_text)


def make_srt(segments, probas, write_probas = True, verbose = False,
             include_preds = True):
    """
    Format segment data with speaker predictions for an .srt file.

    Args:
        segments: A DataFrame containing segment data in each row.
        probas: A DataFrame with speaker probabilities for each segment.
        write_probas: A bool indicating whether to write speaker probabilities
                      to the .srt file instead of speaker names.
    
    Returns:
        A list of strings, each a line for the .srt file.
    """

    srt_text = []
    names = (None if write_probas else
             db.query_names(probas.columns, verbose = verbose).to_dict())
    for i, row in segments.iterrows():
        start = util.get_time_code(row.start, ms_delimit = ",")
        end = util.get_time_code(row.end, ms_delimit = ",")
        

        srt_text.append("{0}\n".format(i))
        srt_text.append("{0} --> {1}\n".format(start, end))
        if include_preds:
            proba_str = ("" if row.label == "S00"
                         else get_proba_str(probas.loc[row.label].iloc[0], names))
            srt_text.append("{0} {1}\n\n".format(row.label, proba_str))
        else:
            srt_text.append("{0}\n\n".format(row.label))

    return srt_text


def get_proba_str(row_proba, names, n_pred = 5):
    """
    Create a string representation of speaker probabilities by segment.
    
    Args:
        row_proba: A Series of probabilities by Person ID.
        names: A dict of int PIDs to str names.
        n_pred: An int indicating the number of top probabilities to use.
    
    Returns:
        A str containing space-separated Person ID to probability pairs.
    """
    def sort_routine(pair):
        pid, proba = pair
        return (round(-proba, 2), pid)  # order by proba desc, pid asc
    row_proba = row_proba.sort_values(ascending = False).head(n_pred)
    pair_list = sorted(row_proba.iteritems(), key = sort_routine)
    if names:
        return ", ".join(["{0}:{1}".format(pid, names[pid])
                          for pid, _ in pair_list])
    else:
        return " ".join(["P{0} {1:.2f}".format(pid, max(0.01, proba))
                         for pid, proba in pair_list])


if __name__ == "__main__":
    main()
