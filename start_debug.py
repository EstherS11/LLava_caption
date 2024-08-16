# -*- coding: utf-8 -*-
import os
import moxing as mox
import argparse
import time
import sys
from ma_config import MAEnv
def prepare_model():
    mox.file.copy_parallel('s3://doclm/zhaojie/interns/zhangfanrui/llava_modelarts/model/', '/home/ma-user/work/zhangfanrui/model/')
    print('model 准备完成')

prepare_model()
print('okokokokokok')



parser = argparse.ArgumentParser()
parser.add_argument('--deepspeed', type=str, default='./scripts/zero2.json', help='the input data path')
parser.add_argument('--train_url', type=str, default='', help='the input data path')
parser.add_argument('--model_name_or_path', type=str, default='/home/ma-user/work/zhangfanrui/model/llava-v1.5-7b', help='the input data path')
parser.add_argument('--version', type=str, default='v1', help='the input data path')
parser.add_argument('--image_folder', type=str, default='/home/ma-user/work/zhangfanrui/data/image/pretrain_forgery', help='the input data path')
parser.add_argument('--data_path', type=str, default='/home/ma-user/work/zhangfanrui/data/json/qa_pairs_forgery_pretrain.json', help='the input data path')
parser.add_argument('--vision_tower', type=str, default='/home/ma-user/work/zhangfanrui/model/clip-vit-large-patch14-336', help='the input data path')
parser.add_argument('--pretrain_mm_mlp_adapter', type=str, default='', help='the input data path')
parser.add_argument('--mm_projector_type', type=str, default='mlp2x_gelu', help='the input data path')
parser.add_argument('--mm_vision_select_layer', type=int, default=-2, help='world size')
parser.add_argument('--output_dir', type=str, default='', help='the input data path')
parser.add_argument('--image_aspect_ratio', type=str, default='pad', help='the input data path')
parser.add_argument('--num_train_epochs', type=int, default=1, help='world size')
parser.add_argument('--per_device_train_batch_size', type=int, default=2, help='world size')
parser.add_argument('--per_device_eval_batch_size', type=int, default=2, help='world size')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='world size')
parser.add_argument('--save_steps', type=int, default=5000, help='world size')
parser.add_argument('--save_total_limit', type=int, default=1, help='world size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='world size')
parser.add_argument('--weight_decay', type=float, default=0.0, help='world size')
parser.add_argument('--warmup_ratio', type=float, default=0.03, help='world size')
parser.add_argument('--lr_scheduler_type', type=str, default='cosine', help='the input data path')
parser.add_argument('--logging_steps', type=int, default=1, help='the input data path')
parser.add_argument('--tf32', action="store_true", help='the input data path')
parser.add_argument('--model_max_length', type=int, default=2048, help='the input data path')
parser.add_argument('--gradient_checkpointing', action="store_true", help='the input data path')
parser.add_argument('--dataloader_num_workers', type=int, default=4, help='the input data path')
parser.add_argument('--lazy_preprocess', action="store_true", help='the input data path')
parser.add_argument('--fp16', action="store_true", help='the input data path') 
parser.add_argument('--debug', action="store_true", help='the input data path')
parser.add_argument('--lora_enable', action="store_true", help='node rank')
parser.add_argument('--freeze_backbone', action="store_true", help='node rank')
parser.add_argument('--lora_r', type=int, default=0, help='world size')
parser.add_argument('--lora_alpha', type=int, default=0, help='world size')
parser.add_argument('--mm_projector_lr', type=float, default=1e-5, help='world size')
parser.add_argument('--master_port', type=str, default='8524', help='the input data path')
args, unparsed = parser.parse_known_args()

maenv = MAEnv(args)
os.system('which python')
maenv.setup_ma_env()
os.system('pip list')
maenv.prepare_env()
os.system('pip list')

while True:
    time.sleep(60)