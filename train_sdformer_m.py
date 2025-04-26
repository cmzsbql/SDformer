import os
import numpy as np
import json
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_sdformer as eval_trans
from dataset import dataset_VQ
import models.transformer_m as trans

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import time
import random

def train():
    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("gpu",args.gpu)
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)
    ##### ---- Dataloader ---- #####
    train_loader_token = dataset_VQ.DATALoader(args.dataname,
                                             64,#1024 for comparison,#64
                                             window_size=args.window_size,
                                             unit_length=2 ** args.down_t, dataset_type='train')

    net = vqvae.VQVAE(args,
                               args.nb_code,
                               args.code_dim,
                               args.down_t,
                               args.stride_t,
                               args.width,
                               args.depth,
                               args.dilation_growth_rate,
                               args.vq_act,
                               args.vq_norm)
    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net = net.float()
    net = net.cuda()

    train_dataset = []
    nb_used=set({})
    ##### ---- get code ---- #####
    for batch in tqdm(train_loader_token):
        batch = batch.cuda().float()
        target = net.encode(batch).cpu().numpy()
        nb_used = nb_used.union(set(target.reshape(-1)))
        train_dataset.append(target)
    print("#####",target.shape,"######")
    print("The number of used code:",len(nb_used))
    train_dataset = np.concatenate(train_dataset, 0)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


    #### ---- Network ---- #####
    trans_encoder = trans.TSG_Transformer(num_vq=args.nb_code,
                                                  embed_dim=args.embed_dim_gpt,
                                                  block_size=args.block_size,
                                                  num_layers=args.num_layers,
                                                  n_head=args.n_head_gpt,
                                                  drop_out_rate=args.drop_out_rate,
                                                  fc_rate=args.ff_rate)
    if args.resume_trans is not None:
        print('loading transformer checkpoint from {}'.format(args.resume_trans))
        ckpt = torch.load(args.resume_trans, map_location='cpu')
        trans_encoder.load_state_dict(ckpt['trans'], strict=True)

    optimizer = optim.AdamW(
        trans_encoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.5, 0.9)
    )
    trans_encoder=trans_encoder.cuda()
    trans_encoder.train()

    repeat_times=100
    from torch.utils.data import ConcatDataset
    train_dataset = [train_dataset for _ in range(repeat_times)]
    train_dataset = ConcatDataset(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              args.batch_size,
                                              shuffle=True,
                                              num_workers=8,
                                              drop_last = False)
    print("######",len(train_loader))
    train_loader_iter = dataset_VQ.cycle(train_loader)

    ##### ---- Optimization goals ---- #####
    loss_ce = torch.nn.CrossEntropyLoss()

    nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
    right_num = 0
    total_num = 0

    ###---- Evaluating ---- #####
    start_time = time.time()
    best_iter_test, best_ds, writer, logger = eval_trans.evaluation_transformer(args,args.out_dir, train_loader_token, trans_encoder,
                                                                                               net, logger, writer, 0,
                                                                                               best_iter=0,
                                                                                               best_ds=99999)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"First evaluaion time: {elapsed_time} seconds")

    
    # ---- Training ---- #####
    mask_token_id = args.nb_code
    while nb_iter <= args.total_iter:
        batch = next(train_loader_iter)
        batch = batch.to(device)
        target = batch
        bs,len_t = target.shape
        input_index = batch.clone()

        mask_prob = random.uniform(0, 1)
        mask = torch.rand(bs, len_t) < mask_prob

        input_index[mask] = mask_token_id

        cls_pred = trans_encoder(input_index)
        cls_pred = cls_pred.contiguous()

        mask_loss = mask.reshape(-1)
        cls_pred = cls_pred.reshape(-1, cls_pred.shape[-1])[mask_loss]
        target = target.reshape(-1)[mask_loss]
        loss_cls = loss_ce(cls_pred, target)

        probs = torch.softmax(cls_pred.float(), dim=-1)

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)
        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        right_num += (cls_pred_index.flatten(0) == target.flatten(0)).sum().item()
        total_num += target.shape[0]
        avg_loss_cls = avg_loss_cls + loss_cls.item()

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()


        nb_iter += 1
        if nb_iter % args.print_iter == 0 :
            avg_loss_cls = avg_loss_cls / args.print_iter
            avg_acc = right_num * 100 / total_num
            writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
            writer.add_scalar('./ACC/train', avg_acc, nb_iter)
            msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
            logger.info(msg)
            avg_loss_cls = 0.
            right_num = 0
            total_num = 0

        if nb_iter % args.eval_iter == 0:
            start_time = time.time()
            best_iter_test, best_ds, writer, logger = eval_trans.evaluation_transformer(args,args.out_dir,
                                                                                                    train_loader_token,
                                                                                                    trans_encoder,
                                                                                                    net, logger, writer,
                                                                                                    nb_iter=nb_iter,
                                                                                                    best_iter=best_iter_test,
                                                                                                    best_ds=best_ds)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"evaluaion time: {elapsed_time} seconds")

        if nb_iter == args.total_iter:
            msg_final2 = f"Train. Iter {best_iter_test} : , DS. {best_ds:.4f}"
            logger.info(msg_final2)
            exit()
if __name__=="__main__":
    train()

