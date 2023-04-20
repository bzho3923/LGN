import os, sys
# set path


current_dir = os.getcwd()
sys.path.append(current_dir)
import argparse
from src.Dataset.protein_dataset import Protein
from src.Dataset.single_mutant_dataset import SingleMutant
from src.Dataset.multi_mutant_dataset import MultiMutant
from src.Dataset.dataset_utils import NormalizeProtein



# Download protein_dataset or Loading your dataset.
def protein_dataset(args,split):
    dataset_arg = {}
    dataset_arg['root'] = args['protein_dataset']
    dataset_arg['set_length'] = None
    dataset_arg['normal_file'] = None
    dataset_arg['divide_num'] = 1
    dataset_arg['divide_idx'] = 0
    dataset_arg['c_alpha_max_neighbors'] = args["dataset_K"]
    dataset_arg["name"] = 40
    
    dataset = Protein(
        dataset_arg["root"],
        dataset_arg["name"],
        split=split,
        divide_num=dataset_arg["divide_num"],
        divide_idx=dataset_arg["divide_idx"],
        c_alpha_max_neighbors=dataset_arg["c_alpha_max_neighbors"],
        set_length=dataset_arg["set_length"],
        random_sampling=True,
        p=args["p"],
        use_sasa=args["sasa"],
        use_bfactor=args["bfactor"],
        use_dihedral=args["dihedral"],
        use_coordinate=args["coordinate"],
        use_denoise=args["denoise"],
        noise_type=args["noise_type"]
        )
    return dataset

# Download singe_mutantDataset or Loading your dataset.
def single_mutant_dataset(args):
    dataset_arg = {}
    dataset_arg['root'] = args['mutant_dataset']
    dataset_arg['name'] = args["single_mutant_name"]
    dataset_arg['raw_dir'] = args['mutant_dataset']+"/DATASET"
    dataset_arg['set_length'] = None
    dataset_arg['normal_file'] = 'data/63w_k10/Train/mean_attr.pt'
    dataset_arg['replace_graph'] = True
    dataset_arg['replace_process'] = True
    dataset_arg['run_root'] = None
    dataset_arg['graph_name'] = 'graph' ##origin=graph
    dataset_arg['processed_name'] = 'processed' #origin="processed"
    dataset_arg['divide_num'] = 1
    dataset_arg['divide_idx'] = 0
    dataset_arg['c_alpha_max_neighbors'] = args["dataset_K"]
    
    sm_dataset = SingleMutant(
        dataset_arg["root"],
        dataset_arg["name"],
        dataset_arg["raw_dir"],
        divide_num=dataset_arg["divide_num"],
        divide_idx=dataset_arg["divide_idx"],
        c_alpha_max_neighbors=dataset_arg["c_alpha_max_neighbors"],
        pre_transform=NormalizeProtein(filename=dataset_arg["normal_file"]),
        graph_name=dataset_arg["graph_name"],
        processed_name=dataset_arg["processed_name"],
        set_length=dataset_arg["set_length"],
        replace_graph=dataset_arg["replace_graph"],
        replace_process=dataset_arg["replace_process"],
    )
    return sm_dataset

# Download multi_mutantDataset or Loading your dataset.
def multi_mutant_dataset(args):
    dataset_arg = {}
    dataset_arg['root'] = args['mutant_dataset']
    dataset_arg['name'] = args['multi_mutant_name']
    dataset_arg['raw_dir'] = args['mutant_dataset']+"/DATASET"
    dataset_arg['set_length'] = None
    dataset_arg['normal_file'] = 'data/63w_k10/Train/mean_attr.pt'
    """
    #Or you can take it to your weight:
    # forexample:
    dataset_arg['normal_file']='/home/yuguang/xinyexiong/protein/63w_k10/Train/mean_attr.pt'
    """
    dataset_arg['c_alpha_max_neighbors'] = args["dataset_K"]
    mm_dataset = MultiMutant(
        dataset_arg["root"],
        dataset_arg["name"],
        dataset_arg["raw_dir"],
        c_alpha_max_neighbors=dataset_arg["c_alpha_max_neighbors"],
        pre_transform=NormalizeProtein(filename=dataset_arg["normal_file"]),
    )
    return mm_dataset


def get_dataset(args):
    # load protein dataset like CATHs40
    train_dataset = protein_dataset(args, "train")
    val_dataset = protein_dataset(args, "val")
    test_dataset = protein_dataset(args, "test")
    # load single mutation dataset
    sm_dataset = single_mutant_dataset(args)
    
    # load multiple mutation dataset 
    mm_dataset = multi_mutant_dataset(args)
    
    # print info
    print(f"Number of train graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(val_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")
    print(f"Number of mutation graphs in single: {len(sm_dataset)}")
    print(f"Number of mutation graphs in multip: {len(mm_dataset)}")
    print("-"*50)
    return train_dataset, val_dataset, test_dataset, sm_dataset, mm_dataset
    