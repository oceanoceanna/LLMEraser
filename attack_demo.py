import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from model.model_interface import MInterface
from data.data_interface import DInterface
from recommender.A_SASRec_final_bce_llm import SASRec, Caser, GRU
from SASRecModules_ori import *
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging
from pytorch_lightning.loggers import TensorBoardLogger
from pandas.core.frame import DataFrame
import os.path as op

from torch.utils.data import DataLoader
from data.data_interface import TrainCollater

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pandas.core.frame import DataFrame
import os.path as op

def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))
    filename_format = f'{args.ratio}-{args.mode_type}-{{epoch:02d}}-{{metric:.3f}}'
    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath=args.ckpt_dir,
        filename=filename_format,
        save_top_k=-1,
        mode='max',
        save_last=True,
        #train_time_interval=args.val_check_interval
        every_n_epochs=1
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks

def test_on_update(model,data_module,device,test_path):
    model = model.to(device)
    model.eval()
    test_content={"generate":[],
                "real":[],
                "cans":[],}
    for batch in data_module.test_dataloader():
        for key in batch['tokens'].keys():
            batch['tokens'][key] = batch['tokens'][key].to(device)
        batch['seq'] = batch['seq'].to(device)
        batch['cans'] = batch['cans'].to(device)
        batch['len_seq'] = batch['len_seq'].to(device)
        batch['len_cans'] = batch['len_cans'].to(device)
        batch['item_id'] = batch['item_id'].to(device)
        generate_output = model.generate(batch)
        output=[]
        for k,generate in enumerate(generate_output):
            real=batch['correct_answer'][k]
            cans=batch['cans_name'][k]
            generate=generate.strip().split("\n")[0]
            output.append((generate,real,cans))
        for generate,real,cans in output:
            test_content["generate"].append(generate)
            test_content["real"].append(real)
            test_content["cans"].append(cans)
    df=DataFrame(test_content)
    df.to_csv(test_path)
    prediction_valid_ratio,hr=model.calculate_hr1(test_content)
    metric=hr*prediction_valid_ratio
    print('test_prediction_valid', prediction_valid_ratio)
    print('test_hr', hr)
    print('metric', metric)
    return prediction_valid_ratio, hr, metric

def get_gradient(model,dataloader,device):
    optimizers = model.get_optimizers_unlearning()
    optimizers.zero_grad()
    model.train()
    print("calculate vector gradient")
    for batch in dataloader:
        for key in batch['tokens'].keys():
            batch['tokens'][key] = batch['tokens'][key].to(device)
        batch['seq'] = batch['seq'].to(device)
        batch['cans'] = batch['cans'].to(device)
        batch['len_seq'] = batch['len_seq'].to(device)
        batch['len_cans'] = batch['len_cans'].to(device)
        batch['item_id'] = batch['item_id'].to(device)
        output = model(batch)
        batch_loss = output.loss
        batch_loss.backward()
    vector_gradients = []
    for p in model.parameters():
        if p.grad is not None:
            grad_copy = ((p.grad.clone().detach())).flatten()
            vector_gradients.append(grad_copy)
    all_gradients = torch.cat(vector_gradients)
    return all_gradients

def get_vector_b(model,data_module,device):
    bofore_loader = data_module.before_dataloader()
    after_loader = data_module.after_dataloader()
    all_gradients_before = get_gradient(model,bofore_loader,device)
    all_gradients_after = get_gradient(model,after_loader,device)
    all_gradients =  (all_gradients_after - all_gradients_before)/len(data_module.trainset)
    vector_shape = all_gradients_before.shape
    return all_gradients, vector_shape

def cg_process_grad(args,vector_shape,vector_b,device,model,data_module):
    initial_tensor = torch.full(vector_shape, args.x_init, dtype=torch.float32).to(device)
    x = torch.nn.Parameter(initial_tensor)
    optimizer_z = torch.optim.Adam([x],lr = args.x_lr,weight_decay = args.x_adjust)
    optimizer_z.zero_grad()
    optimizers = model.get_optimizers_unlearning()
    optimizers.zero_grad()
    large_data_loader = DataLoader(data_module.trainset,
                            batch_size=args.x_batch, 
                            num_workers=data_module.num_workers, 
                            shuffle=True,
                            drop_last=True,
                            collate_fn=TrainCollater(prompt_list=data_module.prompt_list,llm_tokenizer=data_module.llm_tokenizer,train=True, max_step=data_module.max_steps))
    data_loader = iter(large_data_loader)
    print('begin iter')
    for num in range(args.x_iter):
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints")
        try: 
            batch = next(data_loader)
            for key in batch['tokens'].keys():
                batch['tokens'][key] = batch['tokens'][key].to(device)
            batch['seq'] = batch['seq'].to(device)
            batch['cans'] = batch['cans'].to(device)
            batch['len_seq'] = batch['len_seq'].to(device)
            batch['len_cans'] = batch['len_cans'].to(device)
            batch['item_id'] = batch['item_id'].to(device)
        except:
            data_loader = iter(large_data_loader)
            batch  = next(data_loader)
            for key in batch['tokens'].keys():
                batch['tokens'][key] = batch['tokens'][key].to(device)
            batch['seq'] = batch['seq'].to(device)
            batch['cans'] = batch['cans'].to(device)
            batch['len_seq'] = batch['len_seq'].to(device)
            batch['len_cans'] = batch['len_cans'].to(device)
            batch['item_id'] = batch['item_id'].to(device)
            
        ### calculation of hvps
        model.train()
        output = model(batch)
        batch_loss = output.loss
        optimizers.zero_grad()
        params_with_grad = [param for param in model.parameters() if param.requires_grad]
        grads = torch.autograd.grad(batch_loss, params_with_grad, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        hvp = torch.autograd.grad(flat_grads, params_with_grad, grad_outputs=x, retain_graph=True)
        flat_hvps = torch.cat([p.view(-1) for p in hvp])

        ### calculation of the solution
        lossop = 0.5 * (torch.dot(x, flat_hvps)) - torch.dot(vector_b, x)
        optimizer_z.zero_grad()
        lossop.backward()
        optimizer_z.step()
        print("end_iter{}".format(num))

        ### update parameters 
        print("begin update parameterss")
        original_params = []
        for p in model.parameters():
            if p.requires_grad:
                original_params.append(p.clone().detach())
        vector_to_parameters(x, original_params)

        with torch.no_grad():
            params_with_grad = [p for p in model.parameters() if p.requires_grad]
            for p, c in zip(params_with_grad, original_params):
                p.add_(c)
        print("end update parameters")
        
        ### evaluate
        file_path = op.join(args.output_dir, 'test.csv')
        test_on_update(model,data_module,device,file_path)


def main(args):
    pl.seed_everything(args.seed)
    model = MInterface(**vars(args))
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print("load checkpoints from {}".format(args.ckpt_path))

    data_module = DInterface(llm_tokenizer=model.llama_tokenizer,**vars(args))

    args.max_steps=len(data_module.trainset) * args.max_epochs // (args.accumulate_grad_batches * args.batch_size)
    logger = TensorBoardLogger(save_dir='./log/', name=args.log_dir)
    args.callbacks = load_callbacks(args)
    args.logger = logger
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    trainer = Trainer.from_argparse_args(args)

    if args.auto_lr_find:
        lr_finder=trainer.tuner.lr_find(model=model, datamodule=data_module, min_lr=1e-10, max_lr=1e-3, num_training=100)
        fig=lr_finder.plot(suggest=True)
        fig_path="lr_finder.png"
        fig.savefig(fig_path)
        print("Saving to {}".format(fig_path))
        model.hparams.lr=lr_finder.suggestion()

    pl.seed_everything(args.seed)
    device = torch.device('cuda')
    file_path = op.join(args.output_dir, 'test.csv')
    model.eval()
    
    model.train()
    print('calculating vector...')
    vector_b,vector_shape = get_vector_b(model,data_module,device)
    print('calculating done...')
    cg_process_grad(args,vector_shape,vector_b,device,model,data_module)
    

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = ArgumentParser()

    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--devices', default=-1, type=int)
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--amp_backend', default="native", type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--accumulate_grad_batches', default=8, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)

    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)

    parser.add_argument('--load_best', action='store_true')
    parser.add_argument('--load_dir', default=None, type=str)
    parser.add_argument('--load_ver', default=None, type=str)
    parser.add_argument('--load_v_num', default=None, type=int)

    parser.add_argument('--dataset', default='movielens_data', type=str)
    parser.add_argument('--data_dir', default='data/ref/movielens1m', type=str)
    parser.add_argument('--model_name', default='mlp_projector', type=str)
    parser.add_argument('--loss', default='lm', type=str)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--ckpt_dir', default='./checkpoints/', type=str)
    parser.add_argument('--log_dir', default='movielens_logs', type=str)
    
    parser.add_argument('--rec_size', default=64, type=int)
    parser.add_argument('--padding_item_id', default=1682, type=int)
    parser.add_argument('--llm_path', type=str)
    parser.add_argument('--rec_model_path', default='./rec_model/SASRec_ml1m.pt', type=str)
    parser.add_argument('--prompt_path', default='./prompt/movie/', type=str)
    parser.add_argument('--output_dir', default='./output/', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--rec_embed', default="SASRec", choices=['SASRec', 'Caser','GRU'], type=str)

    parser.add_argument('--aug_prob', default=0.5, type=float)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--metric', default='hr', choices=['hr'], type=str)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--save', default='part', choices=['part', 'all'], type=str)
    parser.add_argument('--cans_num', default=10, type=int)
    parser.add_argument("--ckpt_save_dir", type=str, default='./save')
    # Finetuning
    parser.add_argument('--llm_tuning', default='lora', choices=['lora', 'freeze','freeze_lora'], type=str)
    parser.add_argument('--peft_dir', default=None, type=str)
    parser.add_argument('--peft_config', default=None, type=str)
    parser.add_argument('--lora_r', default=8, type=int)
    parser.add_argument('--lora_alpha', default=32, type=float)
    parser.add_argument('--lora_dropout', default=0.1, type=float)

    # Unlearning
    parser.add_argument('--ratio', default=0.05, type=float)
    parser.add_argument('--mode_type', default='delete', type=str)
    parser.add_argument("--x_lr", type=float, default=0.00001)
    parser.add_argument("--x_init", type=float, default=0.000001)
    parser.add_argument("--x_adjust", type=float, default=0.00001)
    parser.add_argument("--x_iter", type=int, default=20)
    parser.add_argument("--x_batch", type=int, default=1)

    args = parser.parse_args()
    
    if 'movielens' in args.data_dir:
        args.padding_item_id = 1682
    elif 'steam' in args.data_dir:
        args.padding_item_id = 3581
    elif 'lastfm' in args.data_dir:
        args.padding_item_id = 4606

    main(args)
