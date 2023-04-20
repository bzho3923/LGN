import argparse
import json
import warnings
import torch
import os
import sys
import yaml
import resource
import numpy as np
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader as pygDataLoader
from numpy import nan
from scipy.stats import spearmanr
from transformers import logging

from model import GNN_model
from data import multi_mutant_dataset#Testing for multi_mutant_dataset.


# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048 * 100, rlimit[1]))



def all_mutant_test(model,loader,protein_names,result_path,device):
    model.eval()
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    spear_cor = np.zeros(protein_num)
    mutation_num = []

    with torch.no_grad():
        for data in loader:
            ### calculate in model
            graph_data = data.to(device)
            out = model(graph_data).to(device)
            data = data.to(device)
            out = torch.log(softmax(out[:, :20]) + 1e-9)
            ## find protein name
            protein_idx = data.protein_idx
            score_info = data.score_info[0]
            num_mutat = len(score_info)
            true_score = torch.zeros(num_mutat)
            pred_score = torch.zeros(num_mutat)
            mutat_pt_num = torch.zeros(num_mutat,dtype=torch.int64)

            # prepare dataframe
            for mutat_idx in range(num_mutat):
                mutat_info, true_score[mutat_idx] = score_info[mutat_idx]
                mutat_pt_num[mutat_idx] = len(mutat_info)
                for i in range(mutat_pt_num[mutat_idx]):
                    item = mutat_info[i]
                    if int(item[1]) >= out.shape[0]:
                        continue
                    pred_score[mutat_idx] += (
                        out[int(item[1] - 1), int(item[2])]
                        - out[int(item[1] - 1), int(item[0])]
                    ).cpu()
            df_score = torch.cat(
                (
                    true_score.reshape((-1, 1)),
                    pred_score.reshape((-1, 1)),
                ),
                1,
            )
            df_score = pd.DataFrame(
                df_score.numpy(), columns=["true_score", "pred_score"]
            )
            if(protein_names[protein_idx]!='.DS_Store'):#you may not need this one.
                result = pd.read_csv(os.path.join(result_path, protein_names[protein_idx],protein_names[protein_idx] + ".tsv"),sep="\t")
                mutation_num.append(len(result))
                result['ours'] = df_score['pred_score']
                print(result["score"],result['ours'])
                result.to_csv(os.path.join(result_path, protein_names[protein_idx],protein_names[protein_idx] + ".csv"),sep="\t", index=False)
                
            if len(df_score["true_score"]) == 0:
                continue
            spear_cor[protein_idx] = spearmanr(
                df_score["true_score"], df_score["pred_score"]
            ).correlation
            if spear_cor[protein_idx] is nan:
                spear_cor[protein_idx] = 0
    print("-"*40)
    all_mutat_num = sum(mutation_num)
    spear_info = {}
    w_avg = 0
    for i in range(protein_num):
        w_avg += spear_cor[i] * mutation_num[i] / all_mutat_num
        spear_info[protein_names[i]] = {}
        spear_info[protein_names[i]]["spearmanr_score"] = spear_cor[i]
        spear_info[protein_names[i]]["mutant_num"] = mutation_num[i]
        print(f"-> {protein_names[i]}: {spear_cor[i]}; mutant_num: {mutation_num[i]}")
    print(f"multi_avg: {spear_cor.mean()}")
    print(f"multi_w_avg: {w_avg}")
    return spear_info, spear_cor.mean(), w_avg

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p",type=float,default=0.5,
                        help="please select the noiseless probability of labelnoise")
    parser.add_argument("--sasa",action="store_true",
                        help="whether to use the sasa feature")
    parser.add_argument("--bfactor",action="store_true",
                        help="whether to use the bfactor feature")
    parser.add_argument("--dihedral",action="store_true",
                        help="whether to use the dihedral feature")
    parser.add_argument("--coordinate",action="store_true",
                        help="whether to use the coordinate feature")
    parser.add_argument("--denoise",action="store_true",
                        help="whether to ues denoise")
    parser.add_argument("--noise_type",type=str,default="wild",
                        help="what kind of noise adding on protein, either wild or substitute")
    parser.add_argument("--date",type=str,default="Sep_25th",
                        help="date using save the filename")
    parser.add_argument("--gnn_device",type=str,default="cuda:0",
                        help="which gpu to use if any (default: 0)")
    parser.add_argument("--gnn",type=str,default="egnn",
                        help="GNN gin, gin-virtual, or gcn, or gcn-virtual or egnn (default: gin-virtual)")

    #Load_your_model
    parser.add_argument("--model_path", type=str, default=None, help="test model name")

    # dataset 
    parser.add_argument("--dataset_K",type=int,default=10,
                        help="the parameter of KNN which used construct the graph, 10 or 20")

    # Attention: If you have dataset,you can change these with your dataset!
    parser.add_argument("--protein_dataset",type=str,default="data/cath40_k10_dyn_imem",
                        help="main protein dataset")
    parser.add_argument("--mutant_dataset",type=str,default="data/evaluation",
                        help="mutation dataset")
    parser.add_argument("--multi_mutant_name",type=str,default=None,
                        help="name of mutation dataset")
    parser.add_argument("--inference_type",type=str,default="mut",
                        help="mask or mut")
    parser.add_argument("--gnn_config",type=str,default="src/Egnnconfig/egnn.yaml",
                        help="gnn config")
    parser.add_argument("--layer_num",type=int,default=6,
                        help="number of layer")
    parser.add_argument("--score_info",type=str,default=None,
                        help="the model output spearmanr score")
    parser.add_argument("--result_path",type=str,default=None,
                        help="the model output spearmanr score")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()
    opt = vars(args)
    gnn_config=yaml.load(open(opt["gnn_config"]),Loader=yaml.FullLoader)[opt["gnn"]]
    device="cuda:0"

    model=GNN_model(gnn_config,opt)

    """
    if you load model which trained by our model,you can load in this way
    """
    model.load_state_dict(torch.load(opt["model_path"])["model_state_dict"])

    """
    if you load model which trained by another model or weight name is different:
    
    bestmodel_weight='/home/yuguang/loty/src/weight/Sep26_AA+SASA_K=10_p=0.6_sasa=True_bfactor=False_dihedral=False_coordinate=False_weighted_loss=False_denoise=False_lambda1=0.2_lambda2=0.5_embeding=False_use_numlayer=6coeff.pt'
    checkpoint=torch.load(bestmodel_weight,map_location=device)
    model_dict=model.state_dict()#Model_dict_name
    for key,value in checkpoint['model_state_dict'].items():
        model_dict["GNN_model."+key]=checkpoint['model_state_dict'][key]
    model.load_state_dict(model_dict)
    model=model.to(device)
    
    """
    


    final_info = {}
    mm_dataset = multi_mutant_dataset(opt)
    protein_names = mm_dataset.protein_names
    print(f"protein_names: {protein_names}")
    multi_mutant_loader = pygDataLoader(mm_dataset, batch_size=1, shuffle=False)
    print(f"Number of mutation graphs in multiple: {len(mm_dataset)}")
    
    multi_info, coeff_list_mm, coeff_mavg = all_mutant_test(
        model, multi_mutant_loader, protein_names, opt["result_path"] ,device
    )
    final_info["multi"] = multi_info
    final_info["multi_mean"] = coeff_list_mm
    final_info["multi_weight_mean"] = coeff_mavg

    with open(opt["score_info"], "w", encoding='utf-8') as f:
        f.write(json.dumps(final_info))