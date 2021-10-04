#!/usr/bin/python2.7
#
# Cut Video Removal Program

from __future__ import print_function

import glob
import os

import mysql.connector


def main():
    query = "SELECT fileName FROM TT_Cuts;"
    records = query_database(query)
    db_paths = set([path[0] for path in records if "test" not in path[0]])
    fs_paths = set([path for path in glob.glob("/videos/original_files/*.mp4")
                    if "-" not in path and "_" not in os.path.basename(path)])
    
    no_fs = sorted(list(db_paths - fs_paths))
    no_db = sorted(list(fs_paths - db_paths))
    
    print("\nDB entries without video:")
    for path in no_fs:
        if os.path.exists(path):
            raise Exception
        print(path)
    
    print("\nVideos without DB entry:")
    for path in no_db:
        print(path)
        os.remove(path)
   
    if len(no_fs) > 1:
        condition = "\" OR fileName = \"".join(no_fs)
        fmt_str = "UPDATE TT_Cuts SET current = 0 WHERE finalized = 0 AND fileName = \"{0}\";"
        query = fmt_str.format(condition)
        update_database(query, True)
    

def query_database(query, verbose = False):
    """
    Query the project's MySQL database with |query| and return a list of
    tuples, each representing a record in the database matching the query.
    """
    cnx = mysql.connector.connect(user = os.environ["DB_USER"],
                                  password = os.environ["DB_PASS"],
                                  database = os.environ["DB_NAME"],
                                  host = os.environ["DB_HOST"])
    cursor = cnx.cursor()
    if verbose:
        print("querying {0}: {1}".format(os.environ["DB_NAME"], query))
    cursor.execute(query)
    records = [record for record in cursor]
    if verbose:
        print(str(len(records)) + " records retrieved from the database")
    cnx.close()
    return records


def update_database(query, verbose = False):
    """
    Update the project's MySQL database with |query|.
    """
    cnx = mysql.connector.connect(user = os.environ["DB_USER"],
                                  password = os.environ["DB_PASS"],
                                  database = os.environ["DB_NAME"],
                                  host = os.environ["DB_HOST"])
    cursor = cnx.cursor()
    if verbose:
        print("updating {0}: {1}".format(os.environ["DB_NAME"], query))
    cursor.execute(query)
    cnx.commit()
    cnx.close()


if __name__ == "__main__":
    main()
