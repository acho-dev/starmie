from sdd.pretrain import load_checkpoint, inference_on_tables
import torch
import pandas as pd
import numpy as np
import glob
import pickle
import time
import sys
import argparse
from tqdm import tqdm

def extractVectors(dfs, dataFolder, augment, sample, table_order, run_id, singleCol=False):
    ''' Get model inference on tables
    Args:
        dfs (list of DataFrames): tables to get model inference on
        dataFolder (str): benchmark folder name
        augment (str): augmentation operator used in vector file path (e.g. 'drop_cell')
        sample (str): sampling method used in vector file path (e.g. 'head')
        table_order (str): 'column' or 'row' ordered
        run_id (int): used in file path
        singleCol (boolean): is this for single column baseline
    Return:
        list of features for the dataframe
    '''
    if singleCol:
        model_path = "results/%s/model_%s_%s_%s_%dsingleCol.pt" % (dataFolder, augment, sample, table_order,run_id)
    else:
        model_path = "results/%s/model_%s_%s_%s_%d.pt" % (dataFolder, augment, sample, table_order,run_id)
    try:
        # Try with weights_only parameter for newer PyTorch versions
        ckpt = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't support weights_only
        ckpt = torch.load(model_path, map_location=torch.device('cuda'))
    # load_checkpoint from sdd/pretain
    model, trainset = load_checkpoint(ckpt, current_dataset=dataFolder)
    return inference_on_tables(dfs, model, trainset, batch_size=1024)

def get_df(dataFolder):
    ''' Get the DataFrames of each table in a folder
    Args:
        dataFolder: filepath to the folder with all tables
    Return:
        dataDfs (dict): key is the filename, value is the dataframe of that table
    '''
    dataFiles = glob.glob(dataFolder+"/*.csv")
    dataDFs = {}
    for file in dataFiles:
        try:
            df = pd.read_csv(file,lineterminator='\n')
            if len(df) > 1000:
                # get first 1000 rows
                df = df.head(1000)
            filename = file.split("/")[-1]
            dataDFs[filename] = df
        except pd.errors.EmptyDataError:
            print(f"   ⚠️  Skipping empty file: {file}")
            continue
    return dataDFs


if __name__ == '__main__':
    ''' Get the model features by calling model inference from sdd/pretrain
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="santos") # can be 'santos', 'santosLarge', 'tus', 'tusLarge', 'wdc'
    # single-column mode without table context
    parser.add_argument("--single_column", dest="single_column", action="store_true")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--table_order", type=str, default='column')
    parser.add_argument("--save_model", dest="save_model", action="store_true")

    hp = parser.parse_args()

    # START PARAMETER: defining the benchmark (dataFolder), if it is a single column baseline,
    # run_id, table_order, and augmentation operators and sampling method if they are different from default
    dataFolder = hp.benchmark
    isSingleCol = hp.single_column
    if 'santos' in dataFolder or dataFolder == 'wdc':
        ao = 'drop_col'
        sm = 'tfidf_entity'
        if isSingleCol:
            ao = 'drop_cell'
    elif dataFolder == 'tus':
        ao = 'drop_cell'
        sm = 'alphaHead'
    elif dataFolder == 'demo':
        ao = 'drop_col'
        sm = 'head'
    else: # dataFolder = tusLarge or custom datasets
        # Check if this is a known benchmark, otherwise use demo-like parameters
        if dataFolder in ['tusLarge']:
            ao = 'drop_cell'
            sm = 'tfidf_entity'
        else:
            # For custom datasets (including dataset_* and other custom names), use demo parameters
            ao = 'drop_col'
            sm = 'head'

    run_id = hp.run_id
    table_order = hp.table_order
    # END PARAMETER

    # Change the data paths to where the benchmarks are stored
    if dataFolder == 'santos':
        DATAPATH = "data/santos/"
        dataDir = ['query', 'datalake']
    elif dataFolder == 'santosLarge':
        DATAPATH = 'data/santos-benchmark/real-benchmark/'
        dataDir = ['query', 'datalake']
    elif dataFolder == 'tus':
        DATAPATH = 'data/table-union-search-benchmark/small/'
        dataDir = ['santos-query', 'benchmark']
    elif dataFolder == 'tusLarge':
        DATAPATH = 'data/table-union-search-benchmark/large/'
        dataDir = ['query', 'benchmark']
    elif dataFolder == 'wdc':
        DATAPATH = {'query': 'data/wdc/query', 'benchmark': 'data/wdc/0/'}
        dataDir = ['query', 'benchmark']
    elif dataFolder == 'demo':
        DATAPATH = 'data/demo/'
        dataDir = ['query', 'datalake']
    else:
        # Handle dataset IDs like dataset_1569749328695144677
        DATAPATH = f'data/{dataFolder}/'
        # Check if query directory has files, otherwise use tables
        query_path = f'data/{dataFolder}/query'
        tables_path = f'data/{dataFolder}/tables'
        datalake_path = f'data/{dataFolder}/datalake'
        
        import os
        has_query = os.path.exists(query_path) and len(os.listdir(query_path)) > 0
        has_tables = os.path.exists(tables_path) and len(os.listdir(tables_path)) > 0
        
        if has_query and has_tables:
            dataDir = ['query', 'tables']
        elif has_query:
            dataDir = ['query', 'datalake']
        elif has_tables:
            dataDir = ['tables', 'datalake']
        else:
            dataDir = ['datalake']

    inference_times = 0
    # dataDir is the query and data lake
    for dir in dataDir:
        print("//==== ", dir)
        if dataFolder == 'wdc':
            DATAFOLDER = DATAPATH[dir]
        else:
            DATAFOLDER = DATAPATH+dir
        dfs = get_df(DATAFOLDER)
        print("num dfs:",len(dfs))

        dataEmbeds = []
        dfs_totalCount = len(dfs)
        dfs_count = 0

        # Extract model vectors, and measure model inference time
        start_time = time.time()
        cl_features = extractVectors(list(dfs.values()), dataFolder, ao, sm, table_order, run_id, singleCol=isSingleCol)
        inference_times += time.time() - start_time
        print("%s %s inference time: %d seconds" %(dataFolder, dir, time.time() - start_time))
        for i, file in enumerate(dfs):
            dfs_count += 1
            # get features for this file / dataset
            cl_features_file = np.array(cl_features[i])
            dataEmbeds.append((file, cl_features_file))
        if dir == 'santos-query':
            saveDir = 'query'
        elif dir == 'benchmark':
            saveDir = 'datalake'
        elif dir == 'tables':
            saveDir = 'datalake'
        else: saveDir = dir

        if isSingleCol:
            output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d_singleCol.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
        else:
            output_path = "data/%s/vectors/cl_%s_%s_%s_%s_%d.pkl" % (dataFolder, saveDir, ao, sm, table_order, run_id)
        if hp.save_model:
            # Create vectors directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pickle.dump(dataEmbeds, open(output_path, "wb"))
        print("Benchmark: ", dataFolder)
        print("--- Total Inference Time: %s seconds ---" % (inference_times))
