import argparse
import os
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to view pickle files')
    parser.add_argument("-p","--pickle",default=None,help="What pickle file to load",type=str)
    args = parser.parse_args()
    pickle_file = args.pickle

    pickle_file = os.path.realpath(pickle_file)

    file = open(pickle_file,"rb")
    data = pickle.load(file)
    print(data)
