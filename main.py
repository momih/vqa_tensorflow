from model import Model
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=20, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=int, default=0.001, help='initial learning rate')
parser.add_argument('--state', dest='state', type=int, default=1024, help='LSTM hidden state size')
parser.add_argument('--embed', dest='embed', type=int, default=300, help='Embedding vector size')
parser.add_argument('--drop', dest='drop', type=int, default=0.5, help='Dropout probability')
parser.add_argument('--freq', dest='freq', type=int, default=1024, help='How many top answers')
parser.add_argument('--resnet_features', dest='resnet', 
                    default='resnet_ckpt/resnet_v2_152.ckpt', 
                    help='Path to resnet pretrained weights')
parser.add_argument('--project', dest='project', type=bool, 
                    default=False, help='Project text features instead of tile')

args = parser.parse_args()

vqa_model = Model(batch_size = args.bs, 
                  init_lr=args.bs, 
                  state_size=args.state,
                  embedding_size=args.embed, 
                  dropout_prob=args.drop, 
                  most_freq_limit=args.freq,
                  resnet_weights_path=args.resnet,
                  project=args.project)

vqa_model.train(args.epoch)
                    