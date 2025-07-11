import argparse

parser = argparse.ArgumentParser(description="FIDTM")

parser.add_argument("--net", default="DQML", type=str)
parser.add_argument("--task", default="1", type=str)
parser.add_argument("--dataset", type=str, default="ShanghaiA")
parser.add_argument("--results", type=str, default="results") 
parser.add_argument("--workers", type=int, default=8, help="load data workers")
parser.add_argument("--print_freq", type=int, default=200, help="print frequency")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=3000, help="end epoch for training")
parser.add_argument("--pre", type=str, default=None, help="pre-trained model")
parser.add_argument("--crop_size", type=int, default=256, help="crop size for training")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--best_pred", type=int, default=1e5, help="best pred")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=5 * 1e-4, help="weight decay")
parser.add_argument("--preload_data", action="store_false", default=True)
parser.add_argument("--visual", action="store_true", default=False)
parser.add_argument("--video_path", type=str, default=None, help="input video path")
parser.add_argument("--depths", default=0, type=int)
parser.add_argument("--groups", default=0, type=int)
parser.add_argument("--order", default=0, type=int)
parser.add_argument("--flag", default=0, type=int)
parser.add_argument("--mu", default=1, type=int)
parser.add_argument("--mu_prox", default=1, type=int)
parser.add_argument("--temperature", default=0.5, type=float)
parser.add_argument("--del_seed", action="store_true") #true false
# federated learning
parser.add_argument("--numbers", type=int, default=1)
parser.add_argument("--local_ep", type=int, default=1)
parser.add_argument("--local_bs", type=int, default=16)


args = parser.parse_args()
return_args = parser.parse_args()
