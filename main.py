import argparse
import torch
import datetime
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from utils.utils import *
# custom
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.food101
import datasets.sun397
import datasets.ucf101
import datasets.imagenet_r
import datasets.imagenet
import datasets.imagenet_s
import datasets.imagenet_a
import datasets.caltech101
import datasets.cifar
from trainers import FCVPT
import os


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.txt_cls = args.txt_cls
    cfg.gpt_prompts = args.gpt_prompts


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    return cfg


class lossmeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count

def train_txt_cls(args, model):
    optimizer, _, _ = setup_text_training_utils(args, model)
    criteria = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in tqdm(range(args.txt_epochs)):
        loss = model.train_txt_clas(criteria)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.txt_cls_init()

def main_worker(args, model, tr_loader, val_loader):
    # first train text classifier
    train_txt_cls(args, model)
    
    all_acc = list()
    optimizer, scheduler, criteria = setup_training_utils(args, model)
    batch_time = lossmeter()
    data_time = lossmeter()

    print("Init features and logits bank")
    num_sample = len(tr_loader.dataset)
    feat_dim = model.backbone_out_size
    class_num = len(model.classes)
    fea_bank = torch.zeros((num_sample, feat_dim))
    score_bank = torch.zeros((num_sample, class_num)).cuda()
    model.eval()
    with torch.no_grad():
        iter_test = iter(tr_loader)
        for i in tqdm(range(len(tr_loader))):
            data = next(iter_test)
            inputs = data["img"]
            indx = data["index"]
            inputs = torch.stack(inputs)  # two views from dataloader
            inputs = inputs.to(model.device)
            logits, features = model.forward_teacher(inputs[0])
            logits_pt = model.forward_student(inputs[1].float().cuda())
            outputs= F.softmax(logits_pt,-1)
            fea_bank[indx] = features.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()
    print("Init bank over")
    
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        model.eval()
        model.pcl_student.train()
        end = time.time()

        for i, batch in enumerate((tr_loader)):
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)

            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(model.device)
            indx = batch["index"]
            
            optimizer.zero_grad()

            pl, features = model.forward_teacher(input[0])
            out = model.forward_student(input[1].float().cuda())
            
            softmax_out = F.softmax(out, dim=-1)
            pseudo_label = F.softmax(pl, dim=-1)
            pseudo_label = pseudo_label.argmax(dim=1, keepdim=True)
            pseudo_label = pseudo_label.flatten().cuda()
            
            # update bank per-minibatch
            with torch.no_grad():
                output_f_ = features.detach().clone().cpu()
                fea_bank[indx] = output_f_.detach().clone().cpu()
                score_bank[indx] = softmax_out.detach().clone()
                distance = output_f_ @ fea_bank.T
                _, idx_near = torch.topk(distance, dim=-1, largest=True, k= args.K + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                score_near = score_bank[idx_near]  # batch x K x C
                
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, args.K, -1)
            L_pull = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
                        
            mask = torch.ones((input[0].shape[0], input[0].shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = softmax_out.T
            dot_neg = softmax_out @ copy  # batch x batch
            dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
            L_push = torch.mean(dot_neg)
           
            loss_NCC = L_pull + L_push
            loss_FCVPT = criteria(out.squeeze(), pseudo_label)
            
            loss = loss_FCVPT + args.lambda_value * loss_NCC
            if i % args.print_freq == 0:
                print("epoch [{0}/{1}][{2}/{3}]\t"
                      "loss {losses1}\t\t"
                      "loss_FCVPT {losses2}\t\t"
                      "loss_NCC {losses3}\t\t"
                      "lr {lr:.6e}".format(
                          epoch + 1, args.epochs, i + 1, len(tr_loader),
                          losses1=loss.item(),
                          losses2=loss_FCVPT.item(),
                          losses3=loss_NCC.item(),
                          lr=optimizer.param_groups[0]["lr"]))
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f'Evaluation: {epoch}')
        acc = test_prompting(val_loader, model)
        print(f'TOP-1 Accuracy: {acc}')
        all_acc.append(acc)
    print(f'-------------------------------- Best Accuracy: {max(all_acc)} --------------------------------')

def main(args):
    cfg = setup_cfg(args)
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed

    dataset_name = cfg.DATASET.NAME
    setup_txt_epochs(args, dataset_name)

    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    trainer = build_trainer(cfg)
    model = trainer.model
    model.args = args
    test_loader = trainer.test_loader
    train_loader = trainer.train_loader_x

    if args.zero_shot:
        CLIP_Inference(model, test_loader)
    else:
        main_worker(args, model,train_loader, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=7777, help="only positive value enables a fixed seed")
    parser.add_argument("--print_freq", type=int, default=10, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="path to config file for dataset setup")
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify config options using the command-line")
    parser.add_argument('--exp-name', type=str, required=False)
    parser.add_argument('--scheduler', default='cosine')
    parser.add_argument('--scheduler-epochs', type=int, default=15)
    parser.add_argument('--scheduler-gamma', type=float, default=0.3)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--acc-batches', type=int, default=1)
    parser.add_argument('--arch', type=str, default='ViT-B/32', required=False)
    parser.add_argument('--gpt_prompts', action='store_true')
    parser.add_argument('--text_prompts', action='store_true')
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--txt_cls', type=str, default='tap', required=True, choices=['zero_shot', 'LLM', 'CLIP'])
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--txt_epochs', type=int, default=1000)
    parser.add_argument('--K', type=int, default=5, help="K-Nearest-Neighbour")
    parser.add_argument("--lambda_value", type=float, default=1.0)
    parser.add_argument('--logfolder', default='logs', type=str)
    args = parser.parse_args()
    args.mile_stones = None
    main(args)
    