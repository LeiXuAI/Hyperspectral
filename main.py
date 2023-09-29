import numpy as np
from data import loadata, get_content_pixels, createImageCubes, get_all_pixels
from concreteVAE import ConcreteVAE
import argparse
from sklearn.preprocessing import minmax_scale
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import get_optimizer, save_res_4kfolds_cv, score, classification_and_eval
import torch
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
import torch.optim as optim
import logging
from pytorch_model_summary import summary
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--seed", type=int, default=100, help="Random seed")
    parser.add_argument("--data_type", type=str, default="pixel", help="input data type is either pixel or window ")
    parser.add_argument("--optimizer", type=str, default="Adam", help="the type of optimizer")
    parser.add_argument("--dataset", type=str, default="IP", help="IP, SA, PU, KSC")
    parser.add_argument("--model", type=str, default="concreteVAE", help="models")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--start_temperature", type=float, default=1.0, help="start temperature")
    parser.add_argument("--end_temperature", type=float, default=0.001, help="end temperature") 
    parser.add_argument("--input_dim", type=int, default=200, help="number of input bands, KSC:176, PU:103, SA:204, IP:200")
    parser.add_argument("--output_dim", type=int, default=200, help="number of output bands for decoder")
    parser.add_argument("--hidden_dim", type=int, default=[256, 256], help="dimension number of hidden layers")
    parser.add_argument("--selected_num", type=int, default=25, help="number of selected bands")
    parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
     

    args = parser.parse_args()
    return args

def get_data():
    args = parse_args_and_config()
    data, label = loadata(args.dataset, False)
    
    n_row, n_column, n_band = data.shape
    norm_data = minmax_scale(data.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    
    if args.data_type == 'pixel':
        data, label = get_content_pixels(norm_data, label)
    elif args.data_type == 'original':
        data, label = get_all_pixels(norm_data, label) 
    else:
        data, label = createImageCubes(norm_data, label) 
    data = data.astype(np.float32) 
    return data, label

def run():
    args = parse_args_and_config()
    torch.manual_seed(args.seed)    
    data, label = loadata(args.dataset, False)
    n_row, n_column, n_band = data.shape
    norm_data = minmax_scale(data.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
    
    if args.data_type == 'pixel':
        data, label = get_content_pixels(norm_data, label)
    elif args.data_type == 'original':
        data, label = get_all_pixels(norm_data, label) 
    else:
        data, label = createImageCubes(norm_data, label) 

    #loading data
    data = data.astype(np.float32)
    train_dataset = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConcreteVAE(args.input_dim, args.output_dim, args.hidden_dim, args.selected_num)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    model.to(device)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10, 25, 35], gamma=0.1)

    # Set temperature and determine rate for decreasing.
    model.sampled_layer.temperature = args.start_temperature
    r = np.power(args.end_temperature / args.start_temperature,
                    1 / ((len(train_dataset) // args.batch_size) * args.epochs)) 
    
    # For tracking loss.
    best_selected_bands = []
    for epoch in range(args.epochs):
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
        print(f'Epoch {epoch+1}\n'
              f'Training Loss: {train_loss / len(train_dataset)}\n' 
              f'Selected bands: {selected_bands + 1}\n'
              f'penalty: {penalty}\n')
        
        best_selected_bands.append(selected_bands + 1)
    torch.save(model.state_dict(), 'saved_model.pth')
    return  best_selected_bands
            
if __name__ == "__main__":
    args = parse_args_and_config() 
    selected_bands = run()
    x, y = get_data()
    score_dic = classification_and_eval(selected_bands[-1] - 1, x, y) 
    np.savez(f'{args.dataset}_{args.selected_num}_{args.lr}.npz', selected_bands=selected_bands, score=score_dic)
    print(f'Dataset: {args.dataset}. The learning rate is: {args.lr}. hidden dim is: {args.hidden_dim}. The result is: {score_dic}.')