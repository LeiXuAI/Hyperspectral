import yaml
import torch
import logging
import numpy as np
from data import loadata, get_input_data
from concreteVAE import ConcreteVAE
import argparse
from sklearn.preprocessing import minmax_scale

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import get_optimizer, save_res_4kfolds_cv, score, classification_and_eval, create_logger

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import torch.optim as optim
from fvcore.common.config import CfgNode

from pytorch_model_summary import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def parse_args():
    parser = argparse.ArgumentParser(description='band selection')
    parser.add_argument('--cfg', help='configurations for training', type=str, required=True)
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    args = parser.parse_args()
    args.cfg = load_cfg_file(args.cfg)
    return args

def load_cfg_file(cfg_file, cfg_dict={}):
    with open(cfg_file) as f:
        cfg_parameters = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in cfg_parameters.items():
            cfg_dict[k] = v
    return cfg_dict

def run_band_selection(data_cfg, model_cfg, train_cfg):
    ori_data, ori_label = loadata(data_cfg['name'], False)
    n_row, n_column, n_band = ori_data.shape
    norm_data = minmax_scale(ori_data.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    data, label = get_input_data(data_cfg['data_type'], norm_data, ori_label)
    
    #loading data
    data = data.astype(np.float32)
    train_dataset = DataLoader(data, batch_size=train_cfg['batch_size'], shuffle=True, drop_last=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConcreteVAE(data_cfg['input_dim'], data_cfg['output_dim'], 
                        model_cfg['hidden_dim'], data_cfg['selected_num'])
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Number of parameters: {total_params}')
   
    model.to(device)
    optimizer = get_optimizer(model_cfg['optimizer'], model.parameters(), train_cfg['lr'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 25, 35], gamma=0.1)
    
    # Set temperature and determine rate for decreasing.
    model.sampled_layer.temperature = model_cfg['start_temperature']
    r = np.power(model_cfg['end_temperature'] / model_cfg['start_temperature'],
                    1 / ((len(train_dataset) // train_cfg['batch_size']) * train_cfg['epochs'])) 
    
    # For tracking loss.
    best_selected_bands = []
    for epoch in range(train_cfg['epochs']):
        train_loss = 0.0
        for i, x in enumerate(train_dataset):
            optimizer.zero_grad()
            x = x.to(device)
            selected_x, selected_bands, loss, penalty = model(x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            model.sampled_layer.temperature *= r
            writer.add_scalar(f'Loss/train: {loss.item()}', i)  
            
        logger.info(f'Epoch {epoch+1}\n'
                    f'Training Loss: {train_loss / len(train_dataset)}\n' 
                    f'Selected bands: {selected_bands + 1}\n'
                    f'penalty: {penalty}\n')
    
        best_selected_bands.append(selected_bands + 1)
    torch.save(model.state_dict(), 'saved_model.pth')
    return best_selected_bands, data, label 
        
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed) 
    cfg = args.cfg
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    logger = create_logger(root_dir=train_cfg['log_path'])
    
    selected_bands, data, label = run_band_selection(data_cfg, model_cfg, train_cfg)
    score_dic = classification_and_eval(selected_bands[-1]-1, data, label) 
    logger.info(f'The result is: {score_dic}.')