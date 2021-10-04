# Amazon Rekognition Preprocess Library

import os
import re
import subprocess

import mysql.connector
import pandas as pd


def query_database(query, index_col = None, verbose = False,
                   user = None,
                   password = None,
                   database = None,
                   host = None):
    """
    Query Digital Democracy's MySQL database.
    
    Args:
        query: A str representing a MySQL query.
        index_col: A str indicating which attribute to use as an index.
    
    Returns:
        A DataFrame structured like the MySQL query result.
    """
    """The lines below moved from the signature
       because they cause this file from being imported
       if the importing script is run by crontab.
    """
    try:     
        user = user if user else os.environ["DB_USER"]
        password = password if password else os.environ["DB_PASS"]
        database = database if database else os.environ["DB_NAME"]
        host = host if host else os.environ["DB_HOST"]
    except:
        raise ValueError("Failed to get dtabase connection info.")

    cnx = mysql.connector.connect(user = user, password = password,
                                  database = database, host = host)
    if verbose:
        print("Querying {0}: {1}".format(database, query))
    try:
        records = pd.read_sql(query, cnx, index_col = index_col)
    except KeyboardInterrupt:
        records = pd.DataFrame()
        alt_cnx = mysql.connector.connect(user = user, password = password,
                                          database = database, host = host)
        query = """
                SELECT id
                FROM INFORMATION_SCHEMA.PROCESSLIST
                WHERE user = "{0}"
                  AND info NOT LIKE "%PROCESSLIST%"
                ORDER BY time
                LIMIT 1;
                """.format(user)
        query_id = pd.read_sql(query, alt_cnx).id[0]
        cursor = alt_cnx.cursor()
        cursor.execute("KILL QUERY {0:d};".format(query_id))
        cursor.close()
        alt_cnx.close()
        if verbose:
            print("\nQuery terminated")
    finally:
        cnx.close()
    if verbose:
        print(len(records.index), "records retrieved from the database")
    return records


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


def get_milliseconds(time_code):
    """
    Extract the number of milliseconds from a time-coded string. 
    
    Args:
        time_code: A str containing a time code of the format HH:MM:SS_MS,
                   where _ can be either a comma or a period.
    
    Returns:
        An int specifying the number of milliseconds in the time code.
    """
    time_code, milliseconds = re.split("[,.]", time_code)
    split_code = time_code.split(":")
    seconds = ((int(split_code[0]) * 3600) + (int(split_code[1]) * 60) + 
               int(split_code[2]))
    return (seconds * 1000) + int(milliseconds)




class Diarizer:

    @staticmethod
    def diarize(file_id):
        MediaDownloader.acquire(file_id)
        seg_path = os.path.join("seg", file_id + ".seg")
        if not os.path.exists(seg_path):
            jar_path = "lium_spkdiarization-8.4.1.jar"
            args = ["/usr/bin/java", "-Xmx1024m", "-jar", jar_path, 
                    "--fInputMask=" + wav_path, "--sOutputMask=" + seg_path, 
                    "--doCEClustering", file_id.strip("-")]
            subprocess.run(args, stderr = subprocess.DEVNULL)
        if os.path.exists(seg_path):
            if file_id[0] == "-":
                args = ["sed", "-i", "s/{0}/-{0}/".format(file_id[1:]),
                        seg_path]
                subprocess.call(args)
            return Diarizer.to_df(file_id, seg_path)

    @staticmethod
    def to_df(path):
        file_id, ext = os.path.splitext(os.path.basename(path))
        if ext not in [".seg", ".srt"]:
            raise ValueError
        if ext == ".srt":
            #Let's not convert speaker tags to int
            regex = (r"(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> " +
                     r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\n(S\d+)")
        else:
            fmt_str = r"{0} 1 (\d+) (\d+) [F|M|U] [S|T] U S(\d+)\n"
            regex = fmt_str.format(file_id)
        pattern = re.compile(regex)
        with open(path, "r") as fp:
            text = fp.read()
        rows = []
        for match in re.findall(pattern, text):
            if ext == ".srt":
                start = get_milliseconds(match[0]) // 1000
                end = get_milliseconds(match[1]) // 1000
                duration = end - start
            else:
                start = int(match[0]) // 100
                duration = int(match[1]) // 100
            label = match[2]
            for second in range(duration):
                rows.append({"second": start + second, "label": label})
        rows.sort(key = lambda seg: seg["second"])
        seg_df = pd.DataFrame(rows).set_index("second")
        if ext == ".seg":
            seg_df = Diarizer._reset_labels(seg_df)
        return seg_df

    @staticmethod
    def _reset_labels(seg_df):
        for i, label in enumerate(seg_df.label.unique()):
            n = 0 if label == "S0" else i + 1
            seg_df.loc[seg_df.label == label, "label"] = "S{0:02d}".format(n)
        return seg_df




class MediaDownloader:

    @staticmethod
    def acquire(file_id, url = None, audio = False):
        MediaDownloader._download_video_from_s3(file_id, url = url)
        if audio:
            MediaDownloader._convert_to_wav(file_id)

    @staticmethod
    def _download_video_from_s3(file_id, url = None):
        if not url:
            url = ("https://s3-us-west-2.amazonaws.com/" +
                   "videostorage-us-west/videos/{0}/{0}.mp4".format(file_id))
        vid_path = os.path.join("mp4", file_id + ".mp4")
        if not MediaDownloader._exists(vid_path):
            args = ['wget', '-O', vid_path, url]
            subprocess.call(args, stdout = subprocess.DEVNULL)
        if not MediaDownloader._exists(vid_path):
            raise Exception("Video Download Failure for File ID: ", file_id)
        return vid_path

    @staticmethod
    def _convert_to_wav(file_id):
        vid_path = os.path.join("mp4", file_id + ".mp4")
        wav_path = os.path.join("wav", file_id + ".wav")
        if not MediaDownloader._exists(wav_path):
            if not MediaDownloader._is_high_quality(vid_path):
                raise ValueError("Video Quality Insufficient for Diarization")
            args = ["ffmpeg", "-i", vid_path, "-vn", "-f", "wav",
                    "-ar", "16000", "-ac", "1", wav_path]
            subprocess.run(args, stdout = subprocess.DEVNULL)
        if not MediaDownloader._exists(vid_path):
            raise Exception("Audio Conversion Failure for File ID: ", file_id)
        return wav_path

    @staticmethod
    def _exists(path):
        return os.path.exists(path) and os.path.getsize(path) > 0

    @staticmethod
    def _is_high_quality(vid_path):
        pattern = "Video: h264 \(Constrained Baseline\)"
        proc = subprocess.Popen(["ffprobe", vid_path],
                                stderr = subprocess.PIPE,
                                universal_newlines = True)
        _, stderr = proc.communicate()
        return re.search(pattern, stderr) is None
