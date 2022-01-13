import numpy as np
import torch
import torch.nn.functional as F
device = torch.device('cuda:0,1,2,3' if torch.cuda.is_available() else 'cpu')
smooth=1
epsilon = 1e-5
from .hausdorff_loss import HausdorffDTLoss, HausdorffERLoss
import math
from torch import einsum
from torch.autograd import Variable
import cv2
from torch.nn import BCELoss



    
def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)
class GDiceLoss():
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape
        
         # (batch size,class_num,x,y,z)
        shp_y = gt.shape 
        # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)
    
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10)**2
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", net_output, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", net_output) + einsum("bcxy->bc", y_onehot))
        divided: torch.Tensor =  - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

class PenaltyGDiceLoss():
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc
def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)

class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full
def threshold_binarize(x, threshold=0.5):#阈值二值化
    return (x > threshold).type(torch.float32)
def robert_suanzi(img):
    r, c = img.shape
    r_sunnzi = [[-1, -1], [1, 1]]
    for x in range(r):
        for y in range(c):
            if (y + 2 <= c) and (x + 2 <= r):
                imgChild = img[x:x + 2, y:y + 2]
                list_robert = r_sunnzi * imgChild
                img[x, y] = abs(list_robert.sum())  # 求和加绝对值
    return img
def dice_coef2(y_pred,y_true):
    
    y_true1=y_true.view(512,512,2)
    y_true1=y_true1.detach().cpu().numpy()
    #r=y_true1.shape[0]
    #c=y_true1.shape[1]
    r_sunnzi =[[-1, -1],[1, 1]]

    list_robert=np.dot(y_true1,r_sunnzi)
    y_true1=abs(list_robert.sum())

    """
    for i in range(2):
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild =y_true1[x:x + 2, y:y + 2,i]
                    
                    list_robert = r_sunnzi * imgChild
                    y_true1[x,y,i] = abs(list_robert.sum())
    """
    
    y_true1=y_true1.flatten()

    

    
    
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    

    

    #w=1/((y_true).sum())
    y_true1=(y_true1>0)
    if y_true1 is not None:
        criterion = BCELoss()
        l=criterion(y_pred,y_true)
        a=(1+0.3*(abs(y_true-y_pred)).mean())

        
        l=(1+0.3*abs((y_true-y_pred).mean()))*l
        
        
        
        
        
        

        return l
    else:
        criterion = BCELoss()
        l=criterion(y_pred,y_true)
        
        
        return l
def dice_coef3(y_pred,y_true):
    smooth=1.0
    y_true1=y_true.view(512,512,2)
    y_true1=y_true1.detach().cpu().numpy()
    #r=y_true1.shape[0]
    #c=y_true1.shape[1]
    r_sunnzi =[[-1, -1],[1, 1]]

    list_robert=np.dot(y_true1,r_sunnzi)
    y_true1=abs(list_robert.sum())

    """
    for i in range(2):
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild =y_true1[x:x + 2, y:y + 2,i]
                    
                    list_robert = r_sunnzi * imgChild
                    y_true1[x,y,i] = abs(list_robert.sum())
    """
    
    y_true1=y_true1.flatten()

    

    
    
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    

    

    #w=1/((y_true).sum())
    y_true1=(y_true1>0)
    if y_true1 is not None:
        intersection=(y_true*y_pred).sum()*(1+0.3*abs((y_true-y_pred).sum()))
        score=(2.*intersection+smooth)/((y_true).sum()*(1+0.3*abs((y_true-y_pred).sum()))+(y_pred).sum()*(1+0.3*abs((y_true-y_pred).sum()))+smooth)
        score=-torch.log(torch.abs(score))
        criterion = BCELoss()
        l=criterion(y_pred,y_true)
        a=(1+0.6*(abs(y_true-y_pred)).mean())

        
        l=(1+0.6*abs((y_true-y_pred).mean()))*l
        score1=0.7*score+0.3*l

        return score1
    else:
        intersection=(y_true*y_pred).sum()
    
    
        score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
   
        score=-torch.log(torch.abs(score))
        criterion = BCELoss()
        l=criterion(y_pred,y_true)
        score1=0.7*score+0.3*l
        return score1

def dice_coef4(y_pred,y_true):
    smooth=1.0
    y_true1=y_true.view(512,512,2)
    y_true1=y_true1.detach().cpu().numpy()
    #r=y_true1.shape[0]
    #c=y_true1.shape[1]
    r_sunnzi =[[-1, -1],[1, 1]]

    list_robert=np.dot(y_true1,r_sunnzi)
    y_true1=abs(list_robert.sum())

    """
    for i in range(2):
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild =y_true1[x:x + 2, y:y + 2,i]
                    
                    list_robert = r_sunnzi * imgChild
                    y_true1[x,y,i] = abs(list_robert.sum())
    """
    
    y_true1=y_true1.flatten()

    

    
    
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    

    

    #w=1/((y_true).sum())
    y_true1=(y_true1>0)
    if y_true1 is not None:
        intersection=(y_true*y_pred).sum()*4
        score=(2.*intersection+smooth)/((y_true).sum()*2+(y_pred).sum()*2+smooth)
        score=-torch.log(torch.abs(score))

        return score
    else:
        intersection=(y_true*y_pred).sum()
    
    
        score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
   
        score=-torch.log(torch.abs(score))
        return score
def dice_coef5(y_pred,y_true):
    smooth=1.0
    y_true1=y_true.view(512,512,2)
    y_true1=y_true1.detach().cpu().numpy()
    #r=y_true1.shape[0]
    #c=y_true1.shape[1]
    r_sunnzi =[[-1, -1],[1, 1]]

    list_robert=np.dot(y_true1,r_sunnzi)
    y_true1=abs(list_robert.sum())

    """
    for i in range(2):
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild =y_true1[x:x + 2, y:y + 2,i]
                    
                    list_robert = r_sunnzi * imgChild
                    y_true1[x,y,i] = abs(list_robert.sum())
    """
    
    y_true1=y_true1.flatten()

    

    
    
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    

    

    #w=1/((y_true).sum())
    y_true1=(y_true1>0)
    if y_true1 is not None:
        intersection=(y_true*y_pred).sum()*5
        score=(2.*intersection+smooth)/((y_true).sum()*2+(y_pred).sum()*2+smooth)
        score=-torch.log(torch.abs(score))

        return score
    else:
        intersection=(y_true*y_pred).sum()
    
    
        score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
   
        score=-torch.log(torch.abs(score))
        return score  
def dice_coef1(y_pred,y_true):
    """2TP / (2TP + FP + FN)"""
    smooth=1.0
    y_true1=y_true.view(512,512,2)
    y_true1=y_true1.detach().cpu().numpy()
    #r=y_true1.shape[0]
    #c=y_true1.shape[1]
    r_sunnzi =[[-1, -1],[1, 1]]

    list_robert=np.dot(y_true1,r_sunnzi)
    y_true1=abs(list_robert.sum())

    """
    for i in range(2):
        for x in range(r):
            for y in range(c):
                if (y + 2 <= c) and (x + 2 <= r):
                    imgChild =y_true1[x:x + 2, y:y + 2,i]
                    
                    list_robert = r_sunnzi * imgChild
                    y_true1[x,y,i] = abs(list_robert.sum())
    """
    
    y_true1=y_true1.flatten()

    

    
    
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    
    

    

    #w=1/((y_true).sum())
    y_true1=(y_true1>0)
    if y_true1 is not None:
        intersection=(y_true*y_pred).sum()
        score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
        score=-torch.log(torch.abs(score))

        return score
    else:
        intersection=(y_true*y_pred).sum()
    
    
        score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
   
        score=-torch.log(torch.abs(score))
        return score 



    
    #intersection=(y_true*y_pred).sum()
    
    
    #score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
   
    #score=-torch.log(torch.abs(score))

    #bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')

    
    #return 0.6*score+0.4*bce

    


    
        
    

    


    
    
    
    
    
    
    
    
    

    
    
    



def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss
def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)

    return normalized_dtm
def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm
def euclidean_distance(p1, p2):
    '''
    计算两个点的欧式距离
    '''
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

class BBox:
    def __init__(self, x, y, r, b):
        '''
        定义框，左上角及右下角坐标
        '''
        self.x, self.y, self.r, self.b = x, y, r, b
    
    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)
    
    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union
    
    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.r, other.r)
        ymax = min(self.b, other.b)
        xmin = max(self.x, other.x)
        ymin = max(self.y, other.y)
        cross_box = BBox(xmin, ymin, xmax, ymax)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0
        return cross_box.area
    
    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        xmin = min(self.x, other.x)
        ymin = min(self.y, other.y)
        xmax = max(self.r, other.r)
        ymax = max(self.b, other.b)
        return BBox(xmin, ymin, xmax, ymax)
    
    def center_distance(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center, other.center)
    
    def bound_diagonal_distance(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        bound = self.boundof(other)
        return euclidean_distance((bound.x, bound.y), (bound.r, bound.b))
    
    @property
    def center(self):
        return (self.x + self.r) / 2, (self.y + self.b) / 2
    
    @property
    def area(self):
        return self.width * self.height
    
    @property
    def width(self):
        return self.r - self.x #+ 1
    
    @property
    def height(self):
        return self.b - self.y #+ 1
def IoU(a, b):
    return a ^ b

def GIoU(a, b):
    bound_area = a.boundof(b).area
    union_area = a | b
    return IoU(a, b) - (bound_area - union_area) / bound_area
def GICE(y_tred,y_true):
    a=BBox(y_tred[0],y_tred[1],y_pred[2],y_pred[3])
    b=BBox(y_true[0],y_true[1],y_true[2],y_true[3])
    bound_area = a.boundof(b).area
    union_area = a | b
    return dsc(y_pred,y_true)-(bound_area-union_area)/bound_area
def giceloss(y_tred,y_true):

    return 1-GICE(y_tred,y_true)

def DIoU(a, b):
    d = a.center_distance(b)
    c = a.bound_diagonal_distance(b)
    return IoU(a, b) - (d ** 2) / (c ** 2)

def CIoU(a, b):
    v = 4 / (math.pi ** 2) * (math.atan(a.width / a.height) - math.atan(b.width / b.height)) ** 2
    iou = IoU(a, b)
    alpha = v / (1 - iou + v)
    return DIoU(a, b) - alpha * v

def hausdorfferloss_bce(y_pred,y_true):
    criterion = HausdorffERLoss(erosions=2)
    

    l1 = criterion.forward(y_pred,y_true)
    l1.to(device)
    
    bce =F.binary_cross_entropy_with_logits(y_pred,y_true,reduction='mean')
    bce.to(device)
    
    
    return (((l1).to(device)+(bce).to(device)).to(device))

def dsc(y_pred,y_true):
    smooth=1.
    
    #y_pred = torch.sigmoid(y_pred) 
    y_true=y_true.flatten()
    y_pred=y_pred.flatten()
    intersection=(y_true*y_pred).sum()
    score=(2.*intersection+smooth)/((y_true).sum()+(y_pred).sum()+smooth)
    return score
def l1(y_pred,y_true):
    #y_pred = F.sigmoid(y_pred) 
    y_pred = y_pred.to(device).float()
    #y_pred = y_pred.view(-1)
    #y_true = y_true.view(-1)
    loss=(y_true-y_pred).sum()
    return loss.float().to(device)
def L2(y_pred,y_true):
    #y_pred=y_pred.flatten()
    #y_true=y_true.flatten()
    # loss=(np.power((y_pred-y_true),2)).sum().cpu()
    
    loss=(torch.pow((y_pred-y_true),2)).sum()
    return loss.float().to(device)
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    input=input().to(device).float()
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return (loss.sum()).to(device)
def hausdorfferloss_l1(y_pred,y_true):
    y_pred = y_pred.to(device).float()
    criterion = HausdorffERLoss()
    l3= criterion.forward(y_pred,y_true).to(device)
    l2=l1(y_pred,y_true)

    return (l3+l2).mean()
def hausdorfferloss_l2(y_pred,y_true):
    criterion = HausdorffERLoss()
    l1 = criterion.forward(y_pred,y_true).to(device)
    l2=L2(y_pred,y_true)
    return ((l1+l2).mean()).to(device)
def hausdorfferloss_smoothl1(y_pred,y_true):
    y_pred=y_pred.to(device).float()
    criterion = HausdorffERLoss()
    l1= criterion.forward(y_pred,y_true).to(device)
    l2=smooth_l1_loss(y_pred,y_true)
    return l1+l2
def hausdorfferloss_dice(y_pred,y_true):
    criterion = HausdorffERLoss()
    l1 = criterion.forward(y_pred,y_true)
    l1.to(device)
    l2=dice_loss(y_true,y_pred)
    l2.to(device)
    return ((l1.to(device)+l2.to(device)).mean()).to(device)
def hausdorffloss_l2bce(y_pred,y_true):
    criterion = HausdorffERLoss()
    l1 = criterion.forward(y_pred,y_true).to(device)
    l2=L2(y_pred,y_true)
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')

    return ((l1+l2+bce).mean()).to(device)


"""
def dsc(y_pred,y_true):
    smooth=1.
    y_true_f=y_true.flatten()
    y_pred_f=y_pred.flatten()
    intersection=np.sum(y_true_f*y_pred_f)
    score=(2.*intersection+smooth)/(np.sum(y_true_f)+np.sum(y_pred_f)+smooth)
    return score
"""
def bce_iou_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (inter + 1) / (union - inter + 1)

    weighted_bce = (weight * bce).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))
    weighted_iou = (weight * iou).sum(dim=(2, 3)) / weight.sum(dim=(2, 3))

    return (weighted_bce + weighted_iou).mean()


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()
def tversky_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return (1 - Tversky) ** gamma
def tversky_bce_loss(pred, mask, alpha=0.5, beta=0.5, gamma=2):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)       

    #flatten label and prediction tensors
    pred = pred.view(-1)
    mask = mask.view(-1)

    #True Positives, False Positives & False Negatives
    TP = (pred * mask).sum()    
    FP = ((1 - mask) * pred).sum()
    FN = (mask * (1 - pred)).sum()

    Tversky = (TP + 1) / (TP + alpha * FP + beta * FN + 1)  

    return bce + (1 - Tversky) ** gamma

def dice_loss(y_true,y_pred):
    loss=1-dsc(y_true,y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos =np.clip(y_pred, 0, 1)#限制一个array的上下界

#给定一个范围[min, max]，数组中值不在这个范围内的，会被限定为这个范围的边界。如给定范围[0, 1]，数组中元素值小于0的，值会变为0，数组中元素值大于1的，要被更改为1.
    y_pred_neg = 1 - y_pred_pos
    y_pos =np.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp =np.sum(y_pos * y_pred_pos)
    fp =np.sum(y_neg * y_pred_pos)
    fn =np.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall
def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (np.sum(y_neg * y_pred_neg) + smooth) / (np.sum(y_neg) + smooth )
    return tn
def tversky(y_true, y_pred):
    y_true_pos=y_true.flatten()
    y_pred_pos=y_pred.flatten()

    true_pos =np.sum(y_true_pos * y_pred_pos)
    false_neg =np.sum(y_true_pos * (1-y_pred_pos))
    false_pos =np.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)
#def tversky_loss(y_pred, y_true):
 #   return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return np.pow((1-pt_1), gamma)
def log_cosh_dice_loss(y_true,y_pred):
    x=dice_loss(y_true,y_pred)
    return torch.log((torch.exp(x)+torch.exp(-x))/2.0)