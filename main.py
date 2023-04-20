import argparse
import warnings
import torch
import os, time
import sys
import yaml
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pygDataLoader
from numpy import nan
from torch_geometric.data import Batch
from scipy.stats import spearmanr
from transformers import logging


##    Loading-model-params:plot,learning-rate and Batch-Sampler.
from src.utils.draw_utils import plot_model
from src.utils.train_utils import lr_scheduler
from src.utils.data_utils import BatchSampler
from src.utils.utils import print_param_num


##    Loading-model and Data
from model import GNN_model
from data import get_dataset

# set path
current_dir=os.getcwd()
sys.path.append(current_dir)
#ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def train(opt,epoch,loader):
    ### $\lambda1$ and $\lambda2$ in paper.
    lambda1=torch.tensor(opt["lambda1"])
    lambda2=torch.tensor(opt["lambda2"])
    model.train()
    train_loss=0
    train_loss_list=[0,0,0,0,0,0]
    for batch_idx,data_list in enumerate(loader):  # Iterate in batches over the training dataset.
        if opt["warmup"] > 0:
            iteration=(epoch - 1) * len(loader) + batch_idx
            for param_group in optimizer.param_groups:
                schedule_lr = lr_scheduler(iteration, epoch, opt)
                param_group["lr"] = schedule_lr
        if type(data_list) is list:
            y = torch.cat([data_.y for data_ in data_list])
            if opt["sasa"]:
                y1 = torch.cat([data_.y1 for data_ in data_list])
            if opt["bfactor"]:
                y2 = torch.cat([data_.y2 for data_ in data_list])
            if opt["dihedral"]:
                y3 = torch.cat([data_.y3 for data_ in data_list])
                y4 = torch.cat([data_.y4 for data_ in data_list])
                y5 = torch.cat([data_.y5 for data_ in data_list])
                y6 = torch.cat([data_.y6 for data_ in data_list])
            if opt["coordinate"]:
                y7 = torch.cat([data_.y7 for data_ in data_list]).to(device)
        else:
            y = data_list.y
            if opt["sasa"]:
                y1 = data_list.y1
            if opt["bfactor"]:
                y2 = data_list.y2
            if opt["dihedral"]:
                y3 = data_list.y3
                y4 = data_list.y4
                y5 = data_list.y5
                y6 = data_list.y6
            if opt["coordinate"]:
                y7 = data_list.y7
        batch_graph=Batch.from_data_list(data_list).to(device)
        out=model(batch_graph).to(device)
        
        loss1=criterion(out[:,:20],y.to(out.device))
        train_loss_list[1]+=loss1.item()
        loss=loss1
        dimention=20
        if opt["sasa"]:
            loss2 = criterion2(out[:, dimention], y1.to(out.device))
            train_loss_list[2] += loss2.item()
            loss = loss + lambda1 * loss2
            dimention = dimention + 1
        if opt["bfactor"]:
            loss3 = criterion2(out[:, dimention], y2.to(out.device))
            train_loss_list[3] += loss3.item()
            loss = loss + lambda1 * loss3
            dimention = dimention + 1
        if opt["dihedral"]:
            loss4 = criterion2(out[:, dimention], y3.to(out.device))
            loss5 = criterion2(out[:, dimention + 1], y4.to(out.device))
            loss6 = criterion2(out[:, dimention + 2], y5.to(out.device))
            loss7 = criterion2(out[:, dimention + 3], y6.to(out.device))
            train_loss_list[4] += (
                loss4.item()+loss5.item()+loss6.item()+loss7.item()
            )
            loss = loss + lambda2 * (loss4 + loss5 + loss6 + loss7)
            dimention = dimention + 4
        if opt["coordinate"]:
            loss8 = criterion2(out[:, dimention : dimention + 3], y7.to(out.device))
            train_loss_list[5] += loss8.item()
            loss = loss + lambda1 * (loss8)
            dimention = dimention + 3
        loss.backward()  # Derive gradients.
        train_loss += loss.item()
        train_loss_list[0] += loss.item()
        if opt["clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),opt["clip"])
        optimizer.step()#Update parameters based on gradients.
        optimizer.zero_grad()#Clear gradients.
        # print("loader lenth:",len(loader))
        if batch_idx % (len(loader) // 2) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(loader),
                    100.0 * batch_idx / len(loader),
                    loss.item(),
                )
            )
    return train_loss / len(loader), train_loss_list


def test(loader,device):
    score_all = torch.tensor([])
    pred_all = torch.tensor([])  # pred label
    label_all = torch.tensor([])
    model.eval()
    correct=0
    test_number=0
    train_loss_list=[0,0,0,0,0,0]
    with torch.no_grad():
        for data_list in loader:  # Iterate in batches over the training/test dataset.
            if type(data_list) is list:
                y = torch.cat([data_.y for data_ in data_list]).to(device)
                label_y = torch.cat([data_.y for data_ in data_list]).to(device)
                if opt["sasa"]:
                    y1 = torch.cat([data_.y1 for data_ in data_list]).to(device)
                if opt["bfactor"]:
                    y2 = torch.cat([data_.y2 for data_ in data_list]).to(device)
                if opt["dihedral"]:
                    y3 = torch.cat([data_.y3 for data_ in data_list]).to(device)
                    y4 = torch.cat([data_.y4 for data_ in data_list]).to(device)
                    y5 = torch.cat([data_.y5 for data_ in data_list]).to(device)
                    y6 = torch.cat([data_.y6 for data_ in data_list]).to(device)
                if opt["coordinate"]:
                    y7 = torch.cat([data_.y7 for data_ in data_list]).to(device)
            else:
                data_list = data_list.to(device)
                y = data_list.y.to(device)
                label_y=data_list.y.to(device)
                if opt["sasa"]:
                    y1 = data_list.y1.to(device)
                if opt["bfactor"]:
                    y2 = data_list.y2.to(device)
                if opt["dihedral"]:
                    y3 = data_list.y3.to(device)
                    y4 = data_list.y4.to(device)
                    y5 = data_list.y5.to(device)
                    y6 = data_list.y6.to(device)
                if opt["coordinate"]:
                    y7 = data_list.y7.to(device)
            batch_graph = Batch.from_data_list(data_list).to(device)
            out = model(batch_graph).to(device)
                
            loss1 = criterion(out[:, :20], y)
            train_loss_list[1] += loss1.item()
            loss = loss1
            dimention = 20
            if opt["sasa"]:
                loss2 = criterion2(out[:, dimention], y1)
                train_loss_list[2] += loss2.item()
                loss = loss + opt["lambda1"] * loss2
                dimention = dimention + 1
            if opt["bfactor"]:
                loss3 = criterion2(out[:, dimention], y2)
                train_loss_list[3] += loss3.item()
                loss = loss + opt["lambda1"] * loss3
                dimention = dimention + 1
            if opt["dihedral"]:
                loss4 = criterion2(out[:, dimention], y3)
                loss5 = criterion2(out[:, dimention + 1], y4)
                loss6 = criterion2(out[:, dimention + 2], y5)
                loss7 = criterion2(out[:, dimention + 3], y6)
                train_loss_list[4] += (
                    loss4.item() + loss5.item() + loss6.item() + loss7.item()
                )
                loss = loss + opt["lambda2"] * (loss4 + loss5 + loss6 + loss7)
                dimention = dimention + 4
            if opt["coordinate"]:
                loss8 = criterion2(out[:, dimention : dimention + 3], y7)
                train_loss_list[5] += loss8.item()
                loss = loss + opt["lambda1"] * (loss8)
                dimention=dimention + 3
            test_number = test_number + len(y)
            out = out[:, :20]
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            score_all = torch.cat((score_all, out.cpu()))
            label_all = torch.cat((label_all, label_y.cpu()))
            pred_all = torch.cat((pred_all, pred.cpu()))
            correct += int((pred == label_y).sum())#Check against ground-truth labels.
    return (
        loss.item(),
        correct / test_number,
        score_all,
        pred_all,
        label_all,
    )  # Derive ratio of correct predictions.


def single_mutant_test(model,loader,protein_names,device):
    model.eval()
    correct = 0
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    score = [torch.tensor([]).to(device) for _ in range(protein_num)]
    probab = [torch.tensor([]).to(device) for _ in range(protein_num)]
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            graph_data=data
            out=model(graph_data).to(device)
            data=data.to(device)
            out = torch.log(softmax(out[: len(data.y), :20]) + 0.000000001)
            probab_ = out - out.gather(1, data.y.reshape((-1, 1)))
            idx = data.protein_idx
            score_ = data.score_matrix[0 : len(data.y)].double()  # note_number*
            score[idx] = torch.concat(
                (score[idx].double(), score_[data.mutat_mask[0 : len(data.y)]])
            )
            probab[idx] = torch.concat(
                (probab[idx], probab_[data.mutat_mask[0 : len(data.y)]])
            )
            pred = out[:, :20].argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            if idx == 1:
                a = torch.zeros(len(out[data.mutat_mask[0 : len(data.y)]]))
                count = 0
                for i in range(data.mutat_mask.shape[0]):
                    for j in range(20):
                        if data.mutat_mask[i, j]:
                            a[count] = out[i, data.y[i]]
                            count += 1
        
    spearman_coeef = np.zeros(protein_num)
    for i in range(protein_num):
        if len(score[i].cpu().numpy()) == 0:
            continue
        spearvalue = spearmanr(
            score[i].cpu().numpy(),probab[i].cpu().numpy()
        ).correlation
        if spearvalue is nan:
            pass
        else:
            spearman_coeef[i] = spearvalue
    
    mutaion_number = sum([len(score[i]) for i in range(protein_num)])
    w_avg = 0
    for i in range(protein_num):
        w_avg += spearman_coeef[i] * len(score[i]) / mutaion_number
    print("-"*40)
    spear_info = {}
    for i in range(protein_num):
        spear_info[protein_names[i]] = spearman_coeef[i]
        print(f"-> {protein_names[i]}: {spearman_coeef[i]}; len: {len(score[i])}")
    print(f"single_avg: {spearman_coeef.mean()}")
    print(f"single_w_avg: {w_avg}")
    # Derive ratio of correct predictions.
    return spear_info, spearman_coeef.mean(), w_avg 



def multi_mutant_test(model,loader,protein_names,device):
    model.eval()
    softmax = nn.Softmax()
    protein_num = len(protein_names)
    spear_cor = np.zeros(protein_num)
    spear_cor_multi = [[] for _ in range(protein_num)]
    row_contents = [[] for _ in range(protein_num)]

    with torch.no_grad():
        for data in loader:
            ### calculate in model
            graph_data = data
            out = model(graph_data).to(device)
            data = data.to(device)
            out = torch.log(softmax(out[:, :20]) + 1e-9)

            ## find protein name
            protein_idx = data.protein_idx
            score_info = data.score_info[0]
            num_mutat = len(score_info)
            true_score = torch.zeros(num_mutat)
            pred_score = torch.zeros(num_mutat)
            mutat_pt_num = torch.zeros(num_mutat, dtype=torch.int64)

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
                    mutat_pt_num.reshape((-1, 1)),
                ),
                1,
            )
            df_score = pd.DataFrame(
                df_score.numpy(), columns=["true_score", "pred_score", "mutat_pt_num"]
            )

            # calculate spearman corr
            if len(df_score["true_score"]) == 0:
                continue
            spear_cor[protein_idx] = spearmanr(
                df_score["true_score"], df_score["pred_score"]
            ).correlation
            if spear_cor[protein_idx] is nan:
                spear_cor[protein_idx] = 0
            spear_cor_multi[protein_idx] = np.zeros(
                (len(df_score["mutat_pt_num"].unique()), 3)
            )
            count = 0
            spear_cor_only_multi = spearmanr(
                df_score[df_score["mutat_pt_num"] != 1]["true_score"],
                df_score[df_score["mutat_pt_num"] != 1]["pred_score"],
            ).correlation

            for mutat_num, group in df_score.groupby("mutat_pt_num"):
                spear_cor_multi[protein_idx][count, 0] = mutat_num
                spear_cor_multi[protein_idx][count, 1] = spearmanr(
                    group["true_score"], group["pred_score"]
                ).correlation
                spear_cor_multi[protein_idx][count, 2] = len(group["true_score"])
                count += 1

            row_contents[protein_idx] = [
                protein_names[protein_idx],
                int(spear_cor_multi[protein_idx][:, 2].sum()),
                spear_cor[protein_idx],
                int(spear_cor_multi[protein_idx][1:, 2].sum()),
                spear_cor_only_multi,
            ]
            for j in range(spear_cor_multi[protein_idx].shape[0]):
                row_contents[protein_idx].append(
                    int(spear_cor_multi[protein_idx][j, 2])
                )
                if spear_cor_multi[protein_idx][j, 1] is nan:
                    row_contents[protein_idx].append("nan")
                else:
                    row_contents[protein_idx].append(spear_cor_multi[protein_idx][j, 1])
    
    print("-"*40)
    all_mutat_num = sum(row_contents[i][1] for i in range(protein_num))
    spear_info = {}
    w_avg = 0
    for i in range(protein_num):
        w_avg += spear_cor[i] * row_contents[i][1] / all_mutat_num
        spear_info[protein_names[i]] = spear_cor[i]
        print(f"-> {protein_names[i]}: {spear_cor[i]}; len: {row_contents[i][1]}")
    
    print(f"multi_avg: {spear_cor.mean()}")
    print(f"multi_w_avg: {w_avg}")
    
    return spear_info, spear_cor.mean(), w_avg


def create_parser():
    parser = argparse.ArgumentParser()

    # train strategy
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
    parser.add_argument("--lambda1",type=float,default=0.2,
                        help="lambda1 in sasa,bfactor,corrdinate loss")
    parser.add_argument("--lambda2",type=float,default=0.5,
                        help="lambda2 in dihedral loss")
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

    # train model
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--init_lr",type=float,default=1e-7,
                        help="init learning rate for warmup")
    parser.add_argument("--warmup",type=int,default=0,
                        help="warm up step")
    parser.add_argument("--weight_decay",type=float,default=1e-2,
                        help="weight_decay")
    parser.add_argument("--num_classes",type=int,default=20,
                        help="number of GNN output (default: 20)")
    parser.add_argument("--epochs",type=int,default=300,
                        help="number of epochs to train (default: 100)")
    parser.add_argument("--step_schedule",type=int,default=350,
                        help="number of epoch schedule lr")
    parser.add_argument("--batch_token_num",type=int,default=4096,
                        help="how many tokens in one batch")
    parser.add_argument("--max_len",type=int,default=4000,
                        help="max token num a graph has")
    parser.add_argument("--node_dim",type=int,default=26,
                        help="number of node feature")
    parser.add_argument("--edge_dim",type=int,default=93,
                        help="number of edge feature")
    parser.add_argument("--layer_num",type=int,default=6,
                        help="number of layer")
    parser.add_argument("--dropout",type=float,default=0,
                        help="dropout rate")
    parser.add_argument("--subtitude_label",action="store_true",
                        help="whether smooth the label by subtitude table")
    parser.add_argument("--JK",type=str,default="last",
                        help="using what nodes embedding to make prediction,last or sum")
    parser.add_argument("--weight_name",type=str,default="",
                        help="which model used to load")
    parser.add_argument("--portion",type=int,default=40,
                        help="mix ratio of af and cath dataset")
    parser.add_argument("--clip",type=float,default=4.0,
                        help="mix ratio of af and cath dataset")
    
    # dataset 
    parser.add_argument("--mix_dataset",action="store_true",
                        help="whether mix alphafold and cath dataset")
    parser.add_argument("--dataset_K",type=int,default=10,
                        help="the parameter of KNN which used construct the graph, 10 or 20")

    #Attention: If you have dataset,you can change these with your dataset!
    parser.add_argument("--protein_dataset",type=str,default="data/cath40_k10_dyn_imem",
                        help="main protein dataset")
    parser.add_argument("--mutant_dataset",type=str,default="data/evaluation",
                        help="mutation dataset")
    parser.add_argument("--single_mutant_name",type=str,default="Evaluation_10",
                        help="name of mutation dataset")
    parser.add_argument("--multi_mutant_name",type=str,default="Evaluation_multi",
                        help="name of mutation dataset")
    parser.add_argument("--inference_type",type=str,default="mut",
                        help="mask or mut")
    parser.add_argument("--gnn_config",type=str,default="src/Egnnconfig/egnn.yaml",
                        help="gnn config")
    args = parser.parse_args()
    return args








if __name__ == "__main__":
    args=create_parser()
    opt=vars(args)



    #config model and cuda.
    gnn_config=yaml.load(open(opt["gnn_config"]),Loader=yaml.FullLoader)[opt["gnn"]]
    ngpus=torch.cuda.device_count()
    device="cuda:0"

    # load dataset and split dataset.
    train_dataset,val_dataset,test_dataset,sm_dataset,mm_dataset=get_dataset(opt)
    def collect_fn(batch):
        return batch
    cath_dataloader=lambda dataset:DataLoader(dataset=dataset,num_workers=5,
                                              collate_fn=lambda x: collect_fn(x),
                                              batch_sampler=BatchSampler(dataset,
                                                                         max_len=opt["max_len"],
                                                                         batch_token_num=opt["batch_token_num"],
                                                                         shuffle=True))
    train_loader,val_loader,test_loader=map(cath_dataloader,(train_dataset,val_dataset,test_dataset))
    protein_names=mm_dataset.protein_names
    single_mutant_loader=pygDataLoader(sm_dataset, batch_size=1, shuffle=False)
    multi_mutant_loader=pygDataLoader(mm_dataset, batch_size=1, shuffle=False)

    # Define file-name and model config details
    filename=(
            opt["date"]
            + f'_feattype={opt["feat_type"]}_K={opt["dataset_K"]}_p={opt["p"]}_sasa={opt["sasa"]}_'
              f'bfactor={opt["bfactor"]}_'
              f'lambda1={opt["lambda1"]}_lambda2={opt["lambda2"]}_'
              f'noise={opt["noise_type"]}_'
              f'gnn={opt["gnn"]}_layer={opt["layer_num"]}_drop={opt["dropout"]}_lr={opt["lr"]}'
    )
    print(filename)
    model=GNN_model(gnn_config,opt)
    print_param_num(model)

    # Define-Loss function and optimizer.
    criterion=torch.nn.CrossEntropyLoss()
    criterion2=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=opt["lr"])
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer,step_size=opt["epochs"] // 2, gamma=0.1
    )

    train_loss_list, val_loss_list, test_loss_list = [], [], []
    single_mean_list, multip_mean_list = [], []
    loss_sum, loss_cla, loss_sas, loss_bfa, loss_dih, loss_cor = [] ,[] ,[] ,[] ,[] ,[]
    
    best_single = -100000
    best_multi = -100000


    #Training.
    for epoch in range(1, opt["epochs"]):
        start = time.time()
        train_loss, loss_list = train(opt,epoch,train_loader)
        train_loss_list.append(train_loss)
        scheduler.step()
        loss_sum.append(loss_list[0] / len(train_loader))
        loss_cla.append(loss_list[1] / len(train_loader))
        loss_sas.append(opt["lambda1"] * loss_list[2] / len(train_loader))
        loss_bfa.append(opt["lambda1"] * loss_list[3] / len(train_loader))
        loss_dih.append(opt["lambda2"] * loss_list[4] / len(train_loader))
        loss_cor.append(opt["lambda1"] * loss_list[5] / len(train_loader))
        if ngpus > 2:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        single_info, single_mean, single_weight = single_mutant_test(
            model, single_mutant_loader, protein_names, device
        )
        multi_info, multi_mean, multi_weight = multi_mutant_test(
            model, multi_mutant_loader, protein_names, device
        )
        single_mean_list.append(single_mean)
        multip_mean_list.append(multi_mean)
        # single correlation achieve best performance
        if single_mean > best_single:
            best_single = single_mean
            torch.save(
                {
                    "args": opt,
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_list": train_loss_list,
                    "test_loss_list": test_loss_list,
                    "mutation_coeff_single": single_mean,
                    "mutation_coeff_multip": multi_mean,
                    "single_info": single_info,
                    "multi_info": multi_info
                },
                "result/weight/" + filename + "coeff.pt",
            )
        # multi correlation achieve best performance
        if multi_mean > best_multi:
            best_multi = multi_mean
        test_loss,test_acc,score_all,pred_all,label_all=test(
            test_loader,device)
        val_loss,_, _, _, _ = test(val_loader,device)
        test_loss_list.append(test_loss)
        val_loss_list.append(val_loss)
        print(
            f"Epoch:{epoch:03d},Train loss:{train_loss:.4f},Test loss:{test_loss:.4f},take{time.time()-start:.2f} s"
        )
        plot_model(
            epoch,
            train_loss_list,
            loss_cla,
            loss_sas,
            loss_bfa,
            loss_dih,
            loss_cor,
            test_loss_list,
            val_loss_list,
            single_mean_list,
            multip_mean_list,
            score_all,
            pred_all,
            label_all,
            filename,
        )#plot model
