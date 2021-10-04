#!/usr/bin/python3
#
# Amazon Rekognition Interface

import argparse
import glob
import json
import os
import sys 
import pickle
import re
import subprocess
import time
import traceback

import boto3
import botocore
import pandas as pd

import rekutil

from os import path


COMMON_DIR = "/home/transcription_tool_common"
BUCKET = "dd-test123"


def main():
    try:
        args = set_options()
        if args.verbose:
            print("Options:", str(vars(args)))

        rek = Rekognizer(args.state, args.url, args.common_dir, keys = args.keys,
                         verbose = args.verbose)
        results = rek.load_data(srt_path = args.srt_path)
        if args.verbose:
            print(results)
        return 0
    except:
        traceback.print_exc(file=sys.stdout)
        return -1


def set_options():
    """
    Retrieve the user-entered arguments for the program.
    """
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: "{0:.3f}".format(x))
    
    parser = argparse.ArgumentParser(description = 
        """Perform Amazon Rekognition on video.""")
    parser.add_argument("state", help = 
        """U.S. state from which video originates""")
    parser.add_argument("-k", "--keys", type=str, nargs = 2,
        metavar = ("ACCESS_KEY", "SECRET_ACCESS_KEY"), help = 
        """AWS access key and secret access key""")
    parser.add_argument("-d", "--db_host", type=str,
        help = 
        """host domain of DigitalDemocracy database""")
    parser.add_argument("-c", "--db_cred", type=str, nargs = 2,
        metavar = ("USER", "PASSWORD"), help = 
        """username and password database credentials""")
    parser.add_argument("-n", "--db_name", type=str,
        help = 
        """name of DigitalDemocracy database to use""")

    parser.add_argument("-r", "--common_dir", type=str,
        default = COMMON_DIR, help = 
        """root directory where related files are stored
           (transcription_tool_common).""")
    parser.add_argument("-b", "--bucket", type=str,
        default = BUCKET, help = 
        """root directory where related files are stored (transcription_tool_common).""")
    parser.add_argument("-v", "--verbose", action = "store_true", help = 
        """print additional information to the terminal as the program is 
           executing""")

    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument("-u", "--url", type=str, help = 
        """location of video to process""")
    group.add_argument("-f", "--file_id", type=str, help = 
        """file ID of video""")
    group.add_argument("-s", "--srt_path", type=str, help = 
        """absolute path of an SRT file to augment Rekognition results""")

    args = parser.parse_args()
    if args.srt_path:
        args.file_id = os.path.splitext(os.path.basename(args.srt_path))[0]
    if args.file_id:
        args.url = ("https://s3-us-west-2.amazonaws.com/" +
                    "{0}/videos/{1}/{1}.mp4".format(args.bucket, args.file_id))
    return args


def get_stored_file_names(bucket_name):
    """
    Return a list of the names of files stored in this object's S3 bucket.
    """
    s3 = boto3.resource("s3")
    file_names = []
    for bucket in s3.buckets.all():
        if bucket.name == bucket_name:
            for obj in bucket.objects.all():
                file_names.append(obj)
    return file_names


def download_images(state, year, directory = "jpeg", db_kwargs = {}):
    """Download the biographical image of every legislator in the given state and
    for the given term year, labeled by their person ID. Store the images in
    the given directory.
    """
    pass
    state = state.upper()
    url = "https://s3-us-west-2.amazonaws.com/dd-drupal-files/images/"
    query = """
            SELECT p.pid, p.image AS image_name
            FROM Person AS p
                 JOIN Term AS t ON p.pid = t.pid
            WHERE t.state = "{0}"
              AND t.year = {1:d};
            """.format(state, year)
    records = rekutil.query_database(query, **db_kwargs)
    records = records[records.image_name.notnull() &
                      records.image_name.str.endswith(".jpg")]
    records = records[~records.image_name.str.contains("/")]
    for _, row in records.iterrows():
        path = os.path.join(directory, state, "{0:08d}.jpg".format(row.pid))
        args = ["wget", url + row.image_name, "-O", path]
        subprocess.call(args)


class Rekognizer:

    def __init__(self, state, url, common_dir, keys = None, verbose = False):
        self.state = state.upper()
        if self.state not in ["CA", "FL", "NY", "TX"]:
            raise ValueError("Invalid State")
        self.role_arn = "arn:aws:iam::492579851961:role/dd-rekognition"
        self.topic_arn = ("arn:aws:sns:us-west-2:492579851961:" +
                          "AmazonRekognitionDigitalDemocracy")
        self.queue_url = ("https://sqs.us-west-2.amazonaws.com/" +
                          "492579851961/dd-rekognition")

        if url:
            region, self.bucket, file_dir, self.file_id = self._parse_url(url)
            self.file_path = file_dir + "/" + self.file_id + ".mp4"
            kwargs = {"region_name": region}
            if keys:
                kwargs.update({"aws_access_key_id": keys[0],
                               "aws_secret_access_key": keys[1]})
            self.rek = boto3.client("rekognition", **kwargs)
            self.sqs = boto3.client("sqs", **kwargs)

        self.verbose = verbose
        self.common_dir = common_dir
        self.root_dir = path.dirname(path.abspath(__file__))

    def get_face_ids(self, aws_kwargs = {}):
        """
        Retrieve previously indexed face IDs from AWS. Return a DataFrame
        mapping person IDs to face IDs.
        """
        response = self.rek.list_faces(CollectionId = self.state)
        ids = [{"pid": os.path.splitext(face["ExternalImageId"])[0],
                "face_id": face["FaceId"]} for face in response["Faces"]]
        return pd.DataFrame(ids)

    def index_images(self, directory, verbose = True):
        """
        Create an AWS face index, which assigns a face ID to each JPEG image
        provided in the given directory.

        Indexed face IDs may be retrieved by calling get_face_ids.
        """
        if self.state not in self.rek.list_collections()["CollectionIds"]:
            response = self.rek.create_collection(self.state)
            if verbose:
                print("Collection ARN:", response["CollectionArn"])
        image_names = glob.glob(os.path.join(directory, "*.jpg"))
        for image_name in image_names:
            image_name = os.path.basename(image_name)
            image_path = os.path.join(self.state, image_name)
            image_id = str(int(os.path.splitext(image_name)[0]))
            kwargs = {"CollectionId": self.state,
                      "Image": {"S3Object": {"Bucket": self.bucket,
                                             "Name": image_path}},
                      "ExternalImageId": image_id,
                      "DetectionAttributes": ["ALL"]}
            response = self.rek.index_faces(**kwargs)
            if verbose:
                print("Faces in", image_name)
                for face_record in response["FaceRecords"]:
                    print(face_record["Face"]["FaceId"])
                print()

    def delete_images(self, save = ()):
        face_ids = self.get_face_ids()
        id_list = face_ids[~face_ids.pid.isin(save)].face_id.values.tolist()
        kwargs = {"CollectionId": self.state, "FaceIds": id_list}
        self.rek.delete_faces(**kwargs)

    def load_data(self, srt_path = None):
        pkl_path = os.path.join(self.root_dir, "pkl", self.file_id + ".pkl")
        if os.path.exists(pkl_path) and srt_path:
            rek_df = pd.read_pickle(pkl_path)
            return self._label_speakers(rek_df, srt_path = srt_path)

        elif os.path.exists(pkl_path):
            print("The rekognition results laready exist!")
            return

        elif srt_path is None:
            results = self._load_results()
            if not results:
                raise ValueError("No Data")
            rek_df = pd.DataFrame(results)
            preds = {}
            for second, group in rek_df.groupby("second"):
                best = 0
                for face_id, id_group in group.groupby("face_id"):
                    conf = id_group.confidence.sum() / len(group)
                    if conf > best:
                        best = conf
                        preds[second] = face_id
            mask = rek_df.apply(lambda row: preds[row.second] == row.face_id,
                                axis = 1)
            rek_df = rek_df[mask]
            rek_df = rek_df.drop_duplicates(["second", "face_id"])
            face_ids = self.get_face_ids()
            rek_df = pd.merge(rek_df, face_ids, on = "face_id")
            rek_df = rek_df.set_index("second", drop = True).sort_index()
            rek_df.to_pickle(pkl_path)

        return

    def _parse_url(self, url):
        """
        Parse the given URL for the AWS region, bucket name, video directory,
        and file ID of the video for Rekognition (in that order). Set this
        object's attributes to the parsed values.
        """
        regex = (r"https://s3-([\w\-]+).amazonaws.com/" +
                 r"([\w\-]+)/([\w\-\/]+?)/(\w+).mp4")
        match = re.match(regex, url)
        if not match:
            raise ValueError("Invalid URL")
        return match.groups()

    def _load_results(self):
        """
        Load archived Rekognition data from a pickled dictionary. If data does
        not exist, make a new Rekognition request.
        """
        dict_path = os.path.join(self.root_dir, "pkl", self.file_id + ".rek")
        if os.path.exists(dict_path):
            with open(dict_path, "rb") as fp:
                results = pickle.load(fp)
        else:
            results = self._run_rekognition()
            with open(dict_path, "wb") as fp:
                pickle.dump(results, fp)
        return results

    def _label_speakers(self, results, srt_path = None):
        try:
            labels = self._load_srt(srt_path = srt_path)
        except FileNotFoundError as e:
            if self.verbose:
                print(e)
            return results
        results = results.join(labels)
        json_path = os.path.join(self.common_dir, "rekognition",
                                 self.file_id + ".json")
        if os.path.exists(json_path):
            with open(json_path, "r") as fp:
                label_dict = json.load(fp)
            return label_dict
        else:
            predictions = []
            for label, group in results.groupby("label"):
                label_dict = {}
                label = "S{0:02d}".format(int(label))
                pid = group.pid.value_counts().index[0]
                label_dict["tag"] = label
                label_dict["pid"] = int(pid)
                predictions.append(label_dict)
            with open(json_path, "w") as fp:
                json.dump(predictions, fp, sort_keys = True)

            return predictions

    def _load_srt(self, srt_path = None):
        if not srt_path:
            srt_path = os.path.join(self.common_dir, "srt",
                                    self.file_id + ".srt")
        if not os.path.exists(srt_path):
            raise FileNotFoundError("\nSRT Not Found: {0}\n".format(srt_path))
        return rekutil.Diarizer.to_df(srt_path)

    def _run_rekognition(self):
        """
        Perform Rekognition on the video corresponding to this object's file ID.
        """
        kwargs = {"Video": {"S3Object": {"Bucket": self.bucket,
                                         "Name": self.file_path}},
                  "CollectionId": self.state,
                  "NotificationChannel": {"RoleArn": self.role_arn,
                                          "SNSTopicArn": self.topic_arn}}
        response = self.rek.start_face_search(**kwargs)
        if self.verbose:
            print("Start Job ID:", response["JobId"])
        while True:
            kwargs = {"QueueUrl": self.queue_url,
                      "MessageAttributeNames": ["ALL"],
                      "MaxNumberOfMessages": 10}
            sqs_response = self.sqs.receive_message(**kwargs)
            if sqs_response and "Messages" in sqs_response:
                for message in sqs_response["Messages"]:
                    notification = json.loads(message["Body"])
                    rek_message = json.loads(notification["Message"])
                    if self.verbose:
                        print("Job ID:", rek_message["JobId"])
                        print("Status:", rek_message["Status"])
                    if str(rek_message["JobId"]) == response["JobId"]:
                        return self._collect_results(rek_message["JobId"])
                    kwargs = {"QueueUrl": self.queue_url,
                              "ReceiptHandle": message["ReceiptHandle"]}
                    self.sqs.delete_message(**kwargs)

    def _collect_results(self, job_id, max_results = 10):
        """
        Read all Rekognition data from the AWS queue as it becomes available
        and store it in a list.
        """
        results = []
        pagination_token = ""
        while True:
            response = self._read_response(job_id, max_results,
                                               pagination_token)
            if self.verbose:
                print("\n\n" + "=" * 40)
            for detected in response["Persons"]:
                for face_match in detected.get("FaceMatches", []):
                    if self.verbose:
                        print("Index:", detected["Person"]["Index"])
                        print("FaceID:", face_match["Face"]["FaceId"])
                        print("Similarity:", face_match["Similarity"])
                        print(self._format_timestamp(detected["Timestamp"]))
                        print("-" * 40)
                    second = round(int(detected["Timestamp"]) / 1000)
                    results.append({"face_id": face_match["Face"]["FaceId"],
                                    "confidence": face_match["Similarity"],
                                    "second": second})
            if "NextToken" not in response:
                break
            pagination_token = response["NextToken"]
        return results

    def _read_response(self, job_id, max_results, pagination_token):
        """
        Continously query the AWS queue until more data is available. Put the
        process to sleep for one second if the request throughput limit is
        exceeded before querying again.
        """
        kwargs = {"JobId": job_id,
                  "MaxResults": max_results,
                  "NextToken": pagination_token,
                  "SortBy": "TIMESTAMP"}
        while True:
            try:
                return self.rek.get_face_search(**kwargs)
            except botocore.exceptions.ClientError:
                # botocore.errorfactory.ProvisionedThroughputExceededException
                traceback.print_exc()
                time.sleep(1)

    def _format_timestamp(self, milliseconds):
        """
        Format the given number of milliseconds into a timestamp string.
        """
        seconds, milliseconds = divmod(milliseconds, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        fmt_str = "Timestamp: {0:02d}:{1:02d}:{2:02d}.{3}"
        return fmt_str.format(hours, minutes, seconds, milliseconds)


if __name__ == "__main__":
    main()
