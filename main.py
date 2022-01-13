import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import argparse
#argparse是python的标准库中用来解析命令行参数的模块，用来替代已经过时的optparse模块，argparse能够根据程序中的定义的sys.argv中解析出这些参数�?#并自动生成帮助和使用信息
import importlib
import sys


def argparsing():
  parser = argparse.ArgumentParser(description='BiO-Net')
  parser.add_argument('--epochs', default=300, type=int, help='trining epochs')#训练轮数
  parser.add_argument('--batch_size', default=2, type=int, help='batch size')#补丁大小
  parser.add_argument('--steps', default=250, type=int, help='steps per epoch')#每轮几步
  parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')#学习�?  
  parser.add_argument('--lr_decay', default=3e-5, type=float, help='learning rate decay')#学习率递减
  parser.add_argument('--num_class', default=1, type=int, help='model output channel number')#输出通道�?  
  parser.add_argument('--multiplier', default=1.0, type=float, help='parameter multiplier')#参数乘数
  parser.add_argument('--iter', default=1, type=int, help='recurrent iteration')#反复迭代
  parser.add_argument('--integrate', action='store_true', help='integrate all inferenced features')#整合所有推断的特征
  parser.add_argument('--save_weight', action='store_true', help='save weight only')#保存权重
  parser.add_argument('--train_data', default='./data/train', type=str, help='data path')
  parser.add_argument('--valid_data', default='./data/valid', type=str, help='data path')
  parser.add_argument('--test_data', default='./data/test', type=str, help='data path')
  parser.add_argument('--exp', default='1', type=str, help='experiment number')#实验数量
  parser.add_argument('--evaluate_only', action='store_true', help='evaluate only?')#只评价？
  parser.add_argument('--save_result', action='store_true', default=True, help='True/False')#保存结果在exp文件�?  parser.add_argument('--model_path', default=None, type=str, help='path to model check')
  parser.add_argument('--model_path', default=None, type=str, help='path to model check')
  parser.add_argument('--valid_dataset', default='monuseg', choices=['monuseg', 'tnbc'], type=str, help='which dataset to validate?')
  parser.add_argument('--backend', default='keras', choices=['keras', 'pytorch'], type=str, help='which backend to use?')
  parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
  parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net/BiONet/DcUnet/unetnetwork')
  parser.add_argument('--lr_weight_decay',type=float,default=1e-4)
  parser.add_argument('--loss_type',type=str,default='bce',help='bce/dice/bce_iou/dice_bce/tversky/hausdorfferloss_bce/hausdorfferloss_l1/hausdorfferloss_smoothl1/hausdorfferloss_dice/hausdorfferloss_l2/hausdorfferloss/hausdorfferloss_l2bce/l2/l2/haousdorfferdt/giceloss')
  args = parser.parse_args()

  print()
  print()
  print(args) # print command line args

  return args
  
def main(args, CORE):
    # path verification路径验证
    if args.model_path is not None:
      if os.path.isfile(args.model_path):
        print('Model path has been verified.')
      else:
        print('Invalid model path! Please specify a valid model file. Program terminating...')
        exit()

    # pipeline starts管道开�?    
    if not args.evaluate_only:
      CORE.train(args)
    CORE.evaluate(args)
    
if __name__ == '__main__':
  # parse command line args
  args = argparsing()

  # import dependencies
  CORE = importlib.import_module(args.backend+'_version')

  main(args, CORE)
