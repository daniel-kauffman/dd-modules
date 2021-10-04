# Query Library

from __future__ import division, print_function
import os
import sys

import mysql.connector
import numpy as np
import pandas as pd


def query_database(query, index_col = None, verbose = False):
    """
    Query Digital Democracy's MySQL database.
    
    Args:
        query: A str representing a MySQL query.
        index_col: A str indicating which attribute to use as an index.
    
    Returns:
        A DataFrame structured like the MySQL query result.
    """
    cnx = mysql.connector.connect(user = os.environ["DB_USER"],
                                  password = os.environ["DB_PASS"],
                                  database = os.environ["DB_NAME"],
                                  host = os.environ["DB_HOST"])
    if verbose:
        print("Querying {0}: {1}".format(os.environ["DB_NAME"], query))
    try:
        records = pd.read_sql(query, cnx, index_col = index_col)
    except KeyboardInterrupt:
        records = pd.DataFrame()
        alt_cnx = mysql.connector.connect(user = os.environ["DB_USER"],
                                          password = os.environ["DB_PASS"],
                                          database = os.environ["DB_NAME"],
                                          host = os.environ["DB_HOST"])
        query = """
                SELECT id
                FROM INFORMATION_SCHEMA.PROCESSLIST
                WHERE user = "{0}"
                  AND info NOT LIKE "%PROCESSLIST%"
                ORDER BY time
                LIMIT 1;
                """.format(os.environ["DB_USER"])
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


def query_pids(file_id, verbose = False):
    """
    Retrieve all Person IDs associated with a File ID.
    
    Args:
        file_id: A str representing a File ID.
    
    Returns:
        A list of int representing Person IDs.
    """
    cids = set(query_cids(file_id, verbose = verbose))
    pids = set(query_pids_by_committees(cids, verbose = verbose) +
               query_authors(file_id, verbose = verbose))
    return sorted(pids)


def query_cids(file_id, verbose = False):
    """
    Query the Committee IDs associated with a File ID.
    
    Args:
        file_id: A str representing a File ID.
    
    Returns:
        A list of int representing Committee IDs.
    
    Raises:
        ValueError: Committee ID does not exist in database.
    """
    query = """
            SELECT DISTINCT h.cid
            FROM CommitteeHearings AS h
                 JOIN Video AS v
                   ON h.hid = v.hid
            WHERE v.fileId = "{0}";
            """.format(file_id)
    records = query_database(query, verbose = verbose)
    cids = [int(cid) for cid in records.cid.values]
    if len(cids) == 0:
        raise ValueError("Invalid Committee ID(s): " + str(cids))
    return cids


def query_pids_by_committees(cids, verbose = False):
    """
    Query the Person IDs belonging to given committees.
    
    Args:
        cids: A list of int representing Committee IDs.
    
    Returns:
        A list of int representing Person IDs.
    
    Raises:
        ValueError: Committee ID has no associated Person IDs.
    """
    query = """
            SELECT DISTINCT pid
            FROM servesOn
            WHERE cid IN ({0})
            ORDER BY pid;
            """.format(", ".join(str(cid) for cid in cids))
    records = query_database(query, verbose = verbose)
    pids = [int(pid) for pid in records.pid.values]
    if len(pids) == 0:
        raise ValueError("Invalid Committee ID(s): " + str(cids))
    return pids


def query_authors(file_id, verbose = False):
    """
    Query bill authors associated with a File ID.
    
    Args:
        file_id: A str representing a File ID.
    
    Returns:
        A list of int representing Person IDs, one for each bill author.
    """
    query = """
            SELECT DISTINCT a.pid
            FROM authors AS a
                 JOIN BillDiscussion AS d
                   ON a.bid = d.bid
                 JOIN Video AS v
                   ON (d.startVideo >= v.vid AND d.endVideo <= v.vid)
            WHERE v.fileId = "{0}";
            """.format(file_id)
    records = query_database(query, verbose = verbose)
    return [int(pid) for pid in records["pid"].tolist()]


def query_legislators(state, year = None, file_id = None, verbose = False):
    """
    Query all legislators by state and session, optionally filtering by only
    those who spoke in specific videos.
    
    Args:
        state: A two-character str representing a U.S. state abbreviation.
        year: An int representing an odd-numbered legislative session.
        file_id: A str representing a File ID.
    
    Returns:
        A list of int representing Person IDs.
    """
    if year is None:
        year = query_current_session(state)
    condition = ("" if file_id is None
                    else "AND v.fileId IN ({0}) ".format(file_id))
    query = """
            SELECT DISTINCT t.pid
            FROM Term AS t
                 JOIN currentUtterance AS u
                   ON t.pid = u.pid
                 JOIN Video AS v
                   ON u.vid = v.vid
            WHERE t.state = "{0}"
              AND t.year = {1:d}
              {2};
            """.format(state, year, condition)
    return sorted(query_database(query, verbose = verbose).pid.values)


def query_names(pids, verbose = False):
    """
    Query the names of speakers by Person IDs (PIDs).
    
    Args:
        pids: A list of int representing Person IDs.
    
    Returns:
        A list of str containing names for the given PIDs.
    """
    query = """
            SELECT pid, CONCAT(LEFT(first, 1), ". ", last) AS name
            FROM Person
            WHERE pid IN ({0});
            """.format(", ".join(str(pid) for pid in pids))
    records = query_database(query, verbose = verbose)
    return records.set_index("pid", drop = True).name


def query_states(verbose = False):
    """
    Query all U.S. states with videos on record.
    
    Returns:
        A list of two-character str, each representing a U.S. state
        abbreviation.
    """
    query = """
            SELECT DISTINCT state
            FROM Video
            WHERE state IS NOT NULL;
            """
    records = query_database(query, verbose = verbose)
    return [str(state) for state in records.state.values]


def query_current_session(state, verbose = False):
    """
    Query the most recent legislative session for the given state.
    
    Args:
        state: A two-character str representing a U.S. state abbreviation.
    
    Returns:
        An int representing the year in which the most recent session began.
    """
    query = """
            SELECT MAX(year) AS year
            FROM Term
            WHERE state = "{0}";
            """.format(state)
    return query_database(query, verbose = verbose).year[0]


def query_utterances(state, year = None, wl_dict = {}, bl_dict = {},
                     shuffle = False, min_len = 1, limit = sys.maxsize,
                     verbose = False):
    """
    Query utterances that satisfy given conditions.
    
    Args:
        state: A two-character str representing a U.S. state abbreviation.
        year: An int representing an odd-numbered legislative session.
        wl_dict: A dict of whitelisted attribute values to include in the query.
                 Supported attributes: hid, file_id, vid, pid, uid
        bl_dict: A dict of blacklisted attribute values to exclude in the query.
                 Supported attributes are the same as wl_dict.
        shuffle: A bool indicating whether to randomize utterances before
                 selecting them.
        min_len: An int indicating the minimum required length of an utterance,
                 in seconds.
        limit: An int restricting the number of utterances returned.
    
    Returns:
        A DataFrame of utterance data, with start and end times in milliseconds.
    """
    if year is None:
        year = query_current_session(state)
    tables = {"hid": "h", "file_id": "v", "vid": "v", "pid": "u", "uid": "u"}
    keys = list(wl_dict.keys()) + list(bl_dict.keys())
    if len(keys) > 0 and not all(key in tables for key in keys):
        raise KeyError("Invalid Table Attribute")
    conditions = []
    for key, attr_list in wl_dict.items():
        fmt_str = "{0}.{1} IN (\"{2}\")"
        str_list = "\", \"".join((str(attr) for attr in attr_list))
        conditions.append(fmt_str.format(tables[key], key, str_list))
    for key, attr_list in bl_dict.items():
        fmt_str = "{0}.{1} NOT IN (\"{2}\")"
        str_list = "\", \"".join((str(attr) for attr in attr_list))
        conditions.append(fmt_str.format(tables[key], key, str_list))
    cond_str = "AND " + " AND ".join(conditions) if conditions else ""
    cond_str = cond_str.replace("file_id", "fileId")
    order_str = "RAND()" if shuffle else "NULL"
    query = """
            SELECT DISTINCT h.hid, v.fileId AS file_id, v.vid, u.pid, u.uid,
                   u.time * 1000 AS start, u.endTime * 1000 AS end,
                   (u.endTime - u.time) * 1000 AS duration,
                   CONVERT(text USING ASCII) AS text
            FROM Hearing AS h
                 JOIN Video AS v
                   ON h.hid = v.hid
                 JOIN currentUtterance AS u
                   ON v.vid = u.vid
            WHERE u.pid IS NOT NULL
              AND h.state = "{0}"
              AND h.session_year = {1:d}
              AND u.endTime - u.time >= {2:d}
              {3}
            ORDER BY {4}
            LIMIT {5:d};
            """.format(state, year, min_len, cond_str, order_str, int(limit))
    records = query_database(query, index_col = "uid", verbose = verbose)
    if len(records) == 0:
        raise ValueError("No Utterances Found")
    return records.drop_duplicates(["pid", "file_id", "start", "end"])
