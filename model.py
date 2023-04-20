import torch
import torch.nn as nn
from src.Module.egnn.network import EGNN

class GNN_model(nn.Module):
    torch.manual_seed(12345)
    
    def __init__(self, gnn_config, opt):
        super().__init__()
       
        # load graph network config which usually not change
        self.gnn_config = gnn_config
        # load global config
        self.opt = opt
        
        if opt.get("layer_num"):
            self.gnn_config["n_layers"] = opt["layer_num"]
        if opt.get("dropout"):
            self.gnn_config["dropout"] = opt["dropout"]
        
        # calculate input dim according to the input feature
        self.out_dim=self._calculate_output_dim()
        self.input_dim = self._calculate_input_dim()
        self.device_count = torch.cuda.device_count()
        # gnn on the rest cudas
        if "egnn" in self.opt["gnn"]:
            self.GNN_model = EGNN(self.gnn_config,self.input_dim,self.out_dim)
        else:
            raise KeyError(f"No implement of {self.opt['gnn']}")
        if self.device_count > 2:
            self.GNN_model = nn.DataParallel(self.GNN_model, device_ids=[i for i in range(1, self.device_count)])
        self.GNN_model=self.GNN_model.to("cuda:1") if self.device_count == 2 else self.GNN_model.to("cuda:0")

    def forward(self, batch_graph):
        gnn_out = self.GNN_model(batch_graph)
        return gnn_out
    
    @torch.no_grad()
    def _calculate_input_dim(self):
        input_size = 31
        return input_size
    
    @torch.no_grad()
    def _calculate_output_dim(self):
        output_dim = 20
        if self.opt["sasa"]:
            output_dim += 1
        if self.opt["bfactor"]:
            output_dim += 1
        if self.opt["dihedral"]:
            output_dim += 4
        if self.opt["coordinate"]:
            output_dim += 3
        return output_dim
