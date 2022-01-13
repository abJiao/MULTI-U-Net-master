from torch.utils.data import DataLoader
import sys
import os
from torch.nn import BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import torch
from matplotlib import pyplot as plt
import imgaug.augmenters as iaa
from tqdm import tqdm
from .metrics import *
from .dataloader import BiONetDataset
from .model import BiONet
from .loss1 import *
#from .modell import *
from .hausdorff_loss import HausdorffDTLoss, HausdorffERLoss
from .network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
from .model2 import *
from .unetnetwork import *
from .res_unet import *
from .unetnetwork1 import *
from .unetnetwork2 import *
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def train(args):
    # augmentations
    transforms = iaa.Sequential([
            iaa.Rotate((-5., 5.)),#随机旋转
            iaa.TranslateX(percent=(-0.05,0.05)),#随机移动
            iaa.TranslateY(percent=(-0.05,0.05)),
            iaa.Affine(shear=(-50, 50)),
            iaa.Affine(scale=(0.8, 1.2)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])

    # load data and create data loaders数据加载器，数据量大
    #train_set = BiONetDataset(args.train_data, 'monuseg', batchsize=args.batch_size, steps=args.steps, transforms=transforms)
    train_set = BiONetDataset(args.train_data, 'monuseg', batchsize=args.batch_size, steps=args.steps,transforms=None)
    test_set = BiONetDataset(args.valid_data, args.valid_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    epochs1=[]
    sval_bce_loss=[]
    sval_bce_iou=[]
    sval_bce_dice=[]
    sval_bce_precision=[]
    
    sval_bce_sensitivity=[]


    sval_dice_iou=[]
    sval_dice_loss=[]
    sval_dice_dice=[]
    sval_dice_precision=[]
    sval_dice_sensitivity=[]

    sval_gbce_iou=[]
    sval_gbce_loss=[]
    sval_gbce_dice=[]
    sval_gbce_precision=[]
    sval_gbce_sensitivity=[]

    sval_gdice_iou=[]
    sval_gdice_loss=[]
    sval_gdice_dice=[]
    sval_gdice_precision=[]
    sval_gdice_sensitivity=[]

    sval_ggdice_iou=[]
    sval_ggdice_loss=[]
    sval_ggdice_dice=[]
    sval_ggdice_precision=[]
    sval_ggdice_sensitivity=[]















    # create model
    '''
    model = BiONet(iterations=args.iter,
                   num_classes=args.num_class,
                   num_layers=4,
                   multiplier=args.multiplier,
                   integrate=args.integrate).to(device).float()
    '''
    if  args.model_type=='U_Net':

        model =U_Net(img_ch=3,output_ch=1)
    elif args.model_type=='R2U_Net':
        model =R2U_Net(img_ch=3, output_ch=1,t=args.t)
    elif args.model_type == 'AttU_Net':
        model =AttU_Net(img_ch=3, output_ch=1)
    elif args.model_type == 'R2AttU_Net':
        model =ResUnet(3,64)
    elif args.model_type=='BiONet':
        model =U_Net3(img_ch=3,output_ch=1)
    elif args.model_type=='DcUnet':
        model=U_Net2(img_ch=3,output_ch=1)
    elif args.model_type=='unetnetwork':
        model=U_Net1(img_ch=3,output_ch=1)
    #if torch.cuda.device_count()>1:

        #model=nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    

    #criterion = BCEWithLogitsLoss()#多标签分类损失 函数
    #if args.loss_type=='bce':
      #  criterion = BCEWithLogitsLoss()
    
        
        
    #criterion=HausdorffERLoss()
    optimizer = Adam(params=model.parameters(), lr=args.lr, weight_decay=args.lr_weight_decay)

    # keras lr decay equivalent  lr递减
    fcn = lambda step: 1./(1. + args.lr_decay * step)
    scheduler = LambdaLR(optimizer, lr_lambda=fcn)#学习率衰减函数，

    print('model successfully built and compiled.')
  
    if not os.path.isdir("checkpoints/"+args.exp):#用于判断某一对象(需提供绝对路径)是否为目录
    	os.mkdir("checkpoints/"+args.exp)#用于以数字权限模式创建目录

    best_iou = 0.
    best_hausdorff=0.
    print('\nStart training...')
    for epoch in range(args.epochs):
        tot_loss = 0.
        tot_iou = 0.
        tot_dice = 0.
        tot_precision=0.
        tot_accuracy=0.
        tot_sensitivity=0.
        
        val_loss = 0.
        val_iou = 0.
        val_dice = 0.
        val_precision=0.
        val_accuracy=0.
        val_sensitivity=0.

        
        
        # training
        model.train()

        for step, (x, y) in enumerate(tqdm(train_loader, desc='[TRAIN] Epoch '+str(epoch+1)+'/'+str(args.epochs))):#Python之tqdm主要作用是用于显示进度，使用较为简单：

#创建进度条；
#关闭进度条；
            if step >= args.steps:
                break
           
            
            x = x.to(device).float()
            
            #将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
            y = y.to(device).float()
            

            optimizer.zero_grad()# 将模型的参数梯度初始化为0
            output = model(x)
            #output= torch.sigmoid(output)
			#SR_flat = SR.view(SR_probs.size(0),-1)

			#GT_flat = GT.view(GT.size(0),-1) 
            
            # 前向传播计算预测值
            


            # loss
            if args.loss_type=='bce':
                criterion = BCELoss()
                l=criterion(output,y)
            elif args.loss_type=='dice':
                l=dice_loss(y,output)
            elif args.loss_type=='bce_iou':
                l=dice_coef2(output,y)
            elif args.loss_type=='dice_bce':
                
                l=dice_coef1(output,y)
            elif args.loss_type=='hausdorfferloss_bce':
            
            
                l=dice_coef3(output,y)
            elif args.loss_type=='hausdorfferloss_l1':
                l=dice_coef4(output,y)
            elif args.loss_type=='hausdorfferloss_l2':
                l=dice_coef5(output,y)
            elif args.loss_type=='hausdorfferloss_smoothl1':
                l=bce1(output,y)
            elif args.loss_type=='hausdorfferloss_dice':
                l=hausdorfferloss_dice(output,y).to(device)
            elif args.loss_type=='hausdorfferloss_l2bce':
                l=hausdorffloss_l2bce(output,y)
            elif args.loss_type=='hausdorfferloss':

                criterion = HausdorffERLoss()
                l= criterion.forward(output,y)

                
            elif args.loss_type=='l2':
                l=L2(output,y)
            elif args.loss_type=='haousdorfferdt':
                criterion =HausdorffDTLoss()
                l= criterion.forward(output,y)
            elif args.loss_type=='giceloss':
                l=giceloss(output,y)
            
                
            #l = criterion(output, y)#损失函数
            #l=criterion.forward(output,y)
            #l = log_cosh_dice_loss(y, output)#损失函数
            

            tot_loss += l.item()#字典内置函数item()可以获取torch.Tensor的值
            #l.requires_grad_(True)
            l.backward() # 反向传播计算梯度 # 反向传播计算梯度
            optimizer.step()#更新所有参数

            # metrics指标
            x, y = output.detach().cpu().numpy(), y.detach().cpu().numpy()
            #datach()阻断反向传播，CPU（）将数据移至cpu,tensor()变量转numpy()
            
            iou_score = iou(y, x)
            precision_score=get_precision(x,y)
            accuracy_score=get_accuracy(x,y)
            sensitivity_score=get_sensitivity(x,y)

            #hausdorff=HausdorffDistance()
            #hausdorff_score=hausdorff.compute(x,y)
            dice_score = dice_coef(y, x)
            
            
            tot_iou += iou_score
            tot_precision+=precision_score
            tot_accuracy+=accuracy_score
            tot_sensitivity+=sensitivity_score

            tot_dice += dice_score
            
            #tot_hausdorff+=hausdorff_score
            
            
            
            
 
            scheduler.step()

        print('[TRAIN] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'loss:', tot_loss/args.steps,
              'iou:', tot_iou/args.steps,
              'dice:', tot_dice/args.steps,
              'precision',tot_precision/args.steps,
              'accuracy',tot_accuracy/args.steps,
              'sensitivity',tot_sensitivity/args.steps
            )
              #'hausdorff',tot_hausdorff/args.steps)
              
        #mean_loss.append(tot_loss/args.steps)
        #mean_dice.append(tot_dice/args.steps)
        # validation
        model.eval()
        
        #如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
        with torch.no_grad():#在eval阶段了，即使不更新，但是在模型中所使用的dropout或者batch norm也就失效了，直接都会进行预测，而使用no_grad则设置让梯度Autograd设置为False(因为在训练中我们默认是True)，这样保证了反向过程为纯粹的测试，而不变参数。

            for step, (x, y) in enumerate(tqdm(test_loader, desc='[VAL] Epoch '+str(epoch+1)+'/'+str(args.epochs))):
                
                x = x.to(device).float()
                y = y.to(device).float()
                

                output = model(x)
                #output= torch.sigmoid(output)

                # loss
                if args.loss_type=='bce':
                    criterion = BCELoss()
                    l=criterion(output,y)
                elif args.loss_type=='dice':
                    l=dice_loss(y,output)
                elif args.loss_type=='bce_iou':
                    l=dice_coef2(output,y)
                elif args.loss_type=='dice_bce':
                    
                    l=dice_coef1(output,y)
                elif args.loss_type=='hausdorfferloss_bce':
                
                
                    l=dice_coef3(output,y)
                elif args.loss_type=='hausdorfferloss_l1':
                    l=dice_coef4(output,y)
                elif args.loss_type=='hausdorfferloss_l2':
                    l=dice_coef5(output,y)
                elif args.loss_type=='hausdorfferloss_smoothl1':
                    l=bce1(output,y)
                elif args.loss_type=='hausdorfferloss_dice':
                    l=hausdorfferloss_dice(output,y).to(device)
                elif args.loss_type=='hausdorfferloss_l2bce':
                    l=hausdorffloss_l2bce(output,y)
                elif args.loss_type=='hausdorfferloss':
                    criterion = HausdorffERLoss()
                    l= criterion.forward(output,y)
                elif args.loss_type=='giceloss':

                    l=giceloss(output,y)
                #l = criterion(output, y)
                #l = criterion.forward(output, y)
                #l = log_cosh_dice_loss(y, output)  # 损失函数
                val_loss +=l.item()

                # metrics
                x, y = output.detach().cpu().numpy(), y.cpu().numpy()
                iou_score = iou(y, x)
                dice_score = dice_coef(y, x)
                precision_score=get_precision(x,y)
                accuracy_score=get_accuracy(x,y)
                sensitivity_score=get_sensitivity(x,y)
                
                #hausdorff_score = HausdorffDistance().compute(x, y)
                val_iou += iou_score
                val_dice += dice_score
                val_precision+=precision_score
                val_accuracy+=accuracy_score
                val_sensitivity+=sensitivity_score
                

                #val_hausdorff += hausdorff_score

        if val_iou/len(test_loader) > best_iou:
            best_iou = val_iou/len(test_loader)
            save_model(args, model)
        """
        if val_hausdorff/len(test_loader)>best_hausdorff:
            best_hausdorff=val_hausdorff/len(test_loader)
            save_model(args,model)
        """
        ssval_loss=val_loss/len(test_loader)
        ssval_iou=val_iou/len(test_loader)
        ssval_dice=val_dice/len(test_loader)
        ssval_precision=val_precision/len(test_loader)
        ssval_accuracy=val_accuracy/len(test_loader)
        ssval_sensitivity=val_sensitivity/len(test_loader)
        

        if args.loss_type=='bce':

            sval_bce_iou.append(ssval_iou)
            sval_bce_loss.append(ssval_loss)
            sval_bce_dice.append(ssval_dice)
            sval_bce_precision.append(ssval_precision)
            sval_bce_sensitivity.append(ssval_sensitivity)
        elif args.loss_type=='dice':
            sval_dice_loss.append(ssval_loss)
            sval_dice_iou.append(ssval_iou)
            sval_dice_dice.append(ssval_dice)
            sval_dice_precision.append(ssval_precision)
            sval_dice_sensitivity.append(ssval_sensitivity)
        elif args.loss_type=='hausdorfferloss_bce':
            sval_gdice_loss.append(ssval_loss)
            sval_gdice_iou.append(ssval_iou)
            sval_gdice_dice.append(ssval_dice)
            sval_gdice_precision.append(ssval_precision)
            sval_gdice_sensitivity.append(ssval_sensitivity)
        elif args.loss_type=='bce_iou':
            sval_gbce_loss.append(ssval_loss)
            sval_gbce_iou.append(ssval_iou)
            sval_gbce_dice.append(ssval_dice)
            sval_gbce_precision.append(ssval_precision)
            sval_gbce_sensitivity.append(ssval_sensitivity)
        elif args.loss_type=='dice_bce':
            sval_ggdice_loss.append(ssval_loss)
            sval_ggdice_iou.append(ssval_iou)
            sval_ggdice_dice.append(ssval_dice)
            sval_ggdice_precision.append(ssval_precision)
            sval_ggdice_sensitivity.append(ssval_sensitivity)

        
        




                    








                    


        


        
        







        print('[VAL] Epoch: '+str(epoch+1)+'/'+str(args.epochs),
              'val_loss:', val_loss/len(test_loader),
              'val_iou:', val_iou/len(test_loader),
              'val_dice:', val_dice/len(test_loader),
              'val_precision',val_precision/len(test_loader),
              'val_accuracy',val_accuracy/len(test_loader),
              'val_sensitivity',val_sensitivity/len(test_loader),
              
             # 'val_housdorff',val_hausdorff/len(test_loader),
             # 'best val_housdorff',best_hausdorff,
              'best val_iou:', best_iou)
        #mean_val_loss.append(val_loss/len(test_loader))
        #mean_val_dice.append(val_dice/len(test_loader))
        epochs1.append(epoch)

    if args.loss_type=='bce':
        print('meiyiiou',sval_bce_iou)
        print('meiyiloss',sval_bce_loss)
        print('meiyidice',sval_bce_dice)
        print('meiyiprecision',sval_bce_precision)
        print('meiyisensitivity',sval_bce_sensitivity)
        plt.plot(epochs1,sval_bce_iou)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('iou')
        plt.savefig('1.png')
    
        plt.plot(epochs1,sval_bce_loss)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('2.png')

        plt.plot(epochs1,sval_bce_dice)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.savefig('3.png')

        plt.plot(epochs1,sval_bce_precision)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('precision')
        plt.savefig('4.png')

        plt.plot(epochs1,sval_bce_sensitivity)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('sensitivity')
        plt.savefig('5.png')
    elif args.loss_type=='dice':
        print('meiyiiou',sval_dice_iou)
        print('meiyiloss',sval_dice_loss)
        print('meiyidice',sval_dice_dice)
        print('meiyiprecision',sval_dice_precision)
        print('meiyisensitivity',sval_dice_sensitivity)
        plt.plot(epochs1,sval_dice_iou)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('iou')
        plt.savefig('1.png')
    
        plt.plot(epochs1,sval_dice_loss)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('2.png')

        plt.plot(epochs1,sval_dice_dice)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.savefig('3.png')

        plt.plot(epochs1,sval_dice_precision)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('precision')
        plt.savefig('4.png')

        plt.plot(epochs1,sval_dice_sensitivity)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('sensitivity')
        plt.savefig('5.png')
    elif args.loss_type=='hausdorfferloss_bce':
        print('meiyiiou',sval_gdice_iou)
        print('meiyiloss',sval_gdice_loss)
        print('meiyidice',sval_gdice_dice)
        print('meiyiprecision',sval_gdice_precision)
        print('meiyisensitivity',sval_gdice_sensitivity)
        plt.plot(epochs1,sval_gdice_iou)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('iou')
        plt.savefig('1.png')
    
        plt.plot(epochs1,sval_gdice_loss)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('2.png')

        plt.plot(epochs1,sval_gdice_dice)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.savefig('3.png')

        plt.plot(epochs1,sval_gdice_precision)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('precision')
        plt.savefig('4.png')

        plt.plot(epochs1,sval_gdice_sensitivity)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('sensitivity')
        plt.savefig('5.png')
    elif args.loss_type=='dice_bce':
        print('meiyiiou',sval_ggdice_iou)
        print('meiyiloss',sval_ggdice_loss)
        print('meiyidice',sval_ggdice_dice)
        print('meiyiprecision',sval_ggdice_precision)
        print('meiyisensitivity',sval_gdice_sensitivity)
        plt.plot(epochs1,sval_ggdice_iou)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('iou')
        plt.savefig('1.png')
    
        plt.plot(epochs1,sval_ggdice_loss)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('2.png')

        plt.plot(epochs1,sval_ggdice_dice)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.savefig('3.png')

        plt.plot(epochs1,sval_ggdice_precision)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('precision')
        plt.savefig('4.png')

        plt.plot(epochs1,sval_ggdice_sensitivity)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('sensitivity')
        plt.savefig('5.png')
    elif args.loss_type=='dice_bce':
        print('meiyiiou',sval_gbce_iou)
        print('meiyiloss',sval_gbcce_loss)
        print('meiyidice',sval_gbcce_dice)
        print('meiyiprecision',sval_gbcce_precision)
        print('meiyisensitivity',sval_gbcce_sensitivity)
        plt.plot(epochs1,sval_gbce_iou)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('iou')
        plt.savefig('1.png')
    
        plt.plot(epochs1,sval_gbce_loss)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig('2.png')

        plt.plot(epochs1,sval_gbce_dice)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('dice')
        plt.savefig('3.png')

        plt.plot(epochs1,sval_gbce_precision)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('precision')
        plt.savefig('4.png')

        plt.plot(epochs1,sval_gbce_sensitivity)
        plt.grid(color='gray', linestyle='--')
        plt.legend(loc=0,ncol=2)
        plt.xlabel('epochs')
        plt.ylabel('sensitivity')
        plt.savefig('5.png')
    
    

    



    



    



    print('\nTraining fininshed!')

def evaluate(args):
    # load data and create data loader
    test_set = BiONetDataset(args.valid_data, args.valid_dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    if args.model_path is None:
        integrate = '_int' if args.integrate else ''
        weights = '_weights'
        cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.pt'
        model_path = "checkpoints/"+args.exp+"/"+cpt_name
    else:
        model_path = args.model_path
    print('Restoring model from path: '+model_path)
    '''
    model = BiONet(iterations=args.iter,
                   num_classes=args.num_class,
                   num_layers=4,
                   multiplier=args.multiplier,
                   integrate=args.integrate).to(device)
    '''
    if args.model_type == 'U_Net':

        model=U_Net(img_ch=3, output_ch=1)
    elif args.model_type == 'R2U_Net':
        model = R2U_Net(img_ch=3, output_ch=1, t=args.t)
    elif args.model_type == 'AttU_Net':
        model = AttU_Net(img_ch=3, output_ch=1)
    elif args.model_type == 'R2AttU_Net':
        model =ResUnet(3,64)
    elif args.model_type=='BiONet':
        model =U_Net3(img_ch=3,output_ch=1)
    elif args.model_type=='DcUnet':
        model=U_Net2(img_ch=3,output_ch=1)
    elif args.model_type=='unetnetwork':
        model=U_Net1(img_ch=3,output_ch=1)
    #if torch.cuda.device_count()>1:

        #model=nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    #criterion = tversky_loss()


    criterion =BCELoss()
    #if args.loss_type=='bce':
     #   criterion = BCEWithLogitsLoss()
    
    #criterion =HausdorffERLoss()
    val_loss = 0.
    val_iou = 0.
    val_dice = 0.
    val_precision=0.
    val_accuracy=0.
    val_sensitivity=0.
    
    segmentations = []
    segmenations1=[]

    # validation
    print('\nStart evaluation...')
    model.eval()


    with torch.no_grad():
        for step, (x, y) in enumerate(tqdm(test_loader)):

            x = x.to(device).float()
            y = y.to(device).float()
            
            output = model(x)
            #output= torch.sigmoid(output)

            # loss
            #l = criterion(output, y)
            if args.loss_type=='bce':
                criterion = BCELoss()
                l=criterion(output,y)
            elif args.loss_type=='dice':
                l=dice_loss(y,output)
            elif args.loss_type=='bce_iou':
                l=dice_coef2(output,y)
            elif args.loss_type=='dice_bce':
                
                l=dice_coef1(output,y)
            elif args.loss_type=='hausdorfferloss_bce':
            
            
                l=dice_coef3(output,y)
            elif args.loss_type=='hausdorfferloss_l1':
                l=dice_coef4(output,y)
            elif args.loss_type=='hausdorfferloss_l2':
                l=dice_coef5(output,y)
            elif args.loss_type=='hausdorfferloss_smoothl1':
                l=bce1(output,y)
            elif args.loss_type=='hausdorfferloss_dice':
                l=hausdorfferloss_dice(output,y).to(device)
            elif args.loss_type=='hausdorfferloss_l2bce':

                l=hausdorffloss_l2bce(output,y)
            elif args.loss_type=='hausdorfferloss':
                criterion = HausdorffERLoss()
                l= criterion.forward(output,y)
            elif args.loss_type=='giceloss':
                
                l=giceloss(output,y)
            #l = criterion.forward(output, y)
            #l = log_cosh_dice_loss(y, output)  # 损失函数
            val_loss +=l.item()
            # metrics
            x, y = output.detach().cpu().numpy(), y.cpu().numpy()
            iou_score = iou(y, x)
            dice_score = dice_coef(y, x)
            precision_score=get_precision(x,y)
            accuracy_score=get_accuracy(x,y)
            sensitivity_score=get_sensitivity(x,y)

            
           # hausdorff_score = HausdorffDistance().compute(x, y)
            val_iou += iou_score
            val_dice += dice_score
            val_precision+=precision_score
            val_accuracy+=accuracy_score
            val_sensitivity+=sensitivity_score

            

            #val_hausdorff += hausdorff_score

            if args.save_result:
                segmentations.append(x)
                segmenations1.append(y)
         
    val_loss = val_loss/len(test_loader)
    val_iou = val_iou/len(test_loader)
    val_dice = val_dice/len(test_loader)
    val_accuracy=val_accuracy/len(test_loader)
    val_precision=val_precision/len(test_loader)
    val_sensitivity=val_sensitivity/len(test_loader)
   
    print('Validation loss:\t', val_loss)
    print('Validation  iou:\t', val_iou)
    print('Validation dice:\t', val_dice)
    print('Validation accuracy:\t', val_accuracy)
    print('Validation precision:\t', val_precision)
    print('Validation sensitivity:\t',val_sensitivity)
   
   # print('Validation hausdorff:\t',val_hausdorff)

    print('\nEvaluation finished!')

    if args.save_result:

        # save metrics
        if not os.path.exists("checkpoints/"+args.exp+"/outputs"):
            os.mkdir("checkpoints/"+args.exp+"/outputs")

        with open("checkpoints/"+args.exp+"/outputs/result.txt", 'w+') as f:
            f.write('Validation loss:\t'+str(val_loss)+'\n')
            f.write('Validation  iou:\t'+str(val_iou)+'\n')
            f.write('Validation dice:\t'+str(val_dice)+'\n')
            #f.write('Validation hausdorff:\t'+str(val_hausdorff)+'\n')
        
        print('Metrics have been saved to:', "checkpoints/"+args.exp+"/outputs/result.txt")

        # save segmentations
        results = np.transpose(np.concatenate(segmentations, axis=0), (0, 2, 3, 1))
        masks=np.transpose(np.concatenate(segmenations1, axis=0), (0, 2, 3, 1))
        results = (results > 0.5).astype(np.float32) # Binarization. Comment out this line if you don't want to

        print('Saving segmentations...')
        if not os.path.exists("checkpoints/"+args.exp+"/outputs/segmentations"):
            os.mkdir("checkpoints/"+args.exp+"/outputs/segmentations")

        for i in range(results.shape[0]):
            plt.imsave("checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+".png",results[i,:,:,0],cmap='gray')
            plt.imsave("checkpoints/"+args.exp+"/outputs/segmentations/"+str(i)+"GT.png",masks[i,:,:,0],cmap='gray') # binary segmenation

        print('A total of '+str(results.shape[0])+' segmentation results have been saved to:', "checkpoints/"+args.exp+"/outputs/segmentations/")

def save_model(args, model):
    integrate = '_int' if args.integrate else ''
    weights = '_weights'
    cpt_name = 'iter_'+str(args.iter)+'_mul_'+str(args.multiplier)+integrate+'_best'+weights+'.pt'
    torch.save({'state_dict':model.state_dict()}, "checkpoints/"+args.exp+"/"+cpt_name)
