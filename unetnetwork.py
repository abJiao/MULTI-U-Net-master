import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv_skip=nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),

            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out)

            )


    def forward(self,x):
        x = self.conv(x)+self.conv_skip(x)
        
        return x

class down_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(down_conv,self).__init__()
        self.down= nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(ch_out),
            #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
    def forward(self,x):
        x=self.down(x)
        return x
"""
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
"""
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up =nn.ConvTranspose2d(
            ch_in, ch_out, kernel_size=4, stride=2,padding=1
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
class U_Net1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv1=down_conv(ch_in=img_ch,ch_out=8)

        
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=8)
        self.down_conv2=down_conv(ch_in=8,ch_out=16)
        self.Conv2 = conv_block(ch_in=16,ch_out=16)
        self.Conv3 = conv_block(ch_in=32,ch_out=32)
        self.down_conv3 = down_conv(ch_in=16,ch_out=32)
        self.Conv4 = conv_block(ch_in=64,ch_out=64)
        self.down_conv4 = down_conv(ch_in=32,ch_out=64)
        self.Conv5 = conv_block(ch_in=128,ch_out=128)
        self.down_conv5 = down_conv(ch_in=64,ch_out=128)
        self.Conv6 = conv_block(ch_in=256,ch_out=256)
        self.down_conv6 = down_conv(ch_in=128,ch_out=256)
        self.Conv7 = conv_block(ch_in=512,ch_out=512)
        self.down_conv7 = down_conv(ch_in=256,ch_out=512)
        self.Conv8 = conv_block(ch_in=1024,ch_out=1024)
        self.down_conv8= down_conv(ch_in=512,ch_out=1024)
        self.Conv9 = conv_block(ch_in=1024,ch_out=2048)
        

        self.Up9 = up_conv(ch_in=2048,ch_out=1024)
        self.Att9 = Attention_block(F_g=1024, F_l=1024, F_int=512)

        self.aux7=nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2,padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv9 = conv_block(ch_in=2048, ch_out=1024)

        self.Up8= up_conv(ch_in=1024,ch_out=512)
        self.Att8 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.aux6 = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv8 = conv_block(ch_in=1024, ch_out=512)
        
        self.Up7 = up_conv(ch_in=512,ch_out=256)
        self.Att7 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.aux5 = nn.Sequential(

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv7 = conv_block(ch_in=512, ch_out=256)

        self.Up6 = up_conv(ch_in=256,ch_out=128)
        self.Att6 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.aux4 = nn.Sequential(


            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

        )
        self.Up_conv6 = conv_block(ch_in=256, ch_out=128)

        self.Up5 = up_conv(ch_in=128,ch_out=64)
        self.Att5 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.aux3 = nn.Sequential(


            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

        )
        self.Up_conv5 = conv_block(ch_in=128, ch_out=64)

        self.Up4 = up_conv(ch_in=64,ch_out=32)
        self.Att4 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.aux2 = nn.Sequential(


            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)

        )
        self.Up_conv4 = conv_block(ch_in=64, ch_out=32)

        self.Up3 = up_conv(ch_in=32,ch_out=16)
        self.Att3 = Attention_block(F_g=16, F_l=16, F_int=8)
        self.aux1 = nn.Sequential(

            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(8, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv3 = conv_block(ch_in=32, ch_out=16)

        self.Up2 = up_conv(ch_in=16,ch_out=8)
        self.Att2 = Attention_block(F_g=8, F_l=8, F_int=4)
        self.Up_conv2 = conv_block(ch_in=16, ch_out=8)

        self.conv=nn.Conv2d(8,output_ch,kernel_size=1,stride=1,padding=0)

        #self.Conv_1x1 = nn.Conv2d(8,output_ch,kernel_size=1,stride=1,padding=0)
        self.output_layer=nn.Sigmoid()



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)
        y1= self.down_conv1(x)

        x2 = self.Maxpool(x1)
        
        x2=torch.cat((y1,x2),dim=1)
        #x2=torch.cat((y1,x2),dim=1)
        
        x2 = self.Conv2(x2)
        y2=self.down_conv2(y1)
        x3 = self.Maxpool(x2)
        x3=torch.cat((y2,x3),dim=1)
        x3 = self.Conv3(x3)
        y3=self.down_conv3(y2)

        x4 = self.Maxpool(x3)
        x4=torch.cat((y3,x4),dim=1)
        x4 = self.Conv4(x4)
        y4=self.down_conv4(y3)

        x5 = self.Maxpool(x4)
        x5=torch.cat((y4,x5),dim=1)
        x5 = self.Conv5(x5)
        y5=self.down_conv5(y4)

        x6 = self.Maxpool(x5)
        x6=torch.cat((y5,x6),dim=1)
        x6 = self.Conv6(x6)
        y6=self.down_conv6(y5)

        x7 = self.Maxpool(x6)
        x7=torch.cat((y6,x7),dim=1)
        x7 = self.Conv7(x7)
        y7=self.down_conv7(y6)

        x8 = self.Maxpool(x7)
        x8=torch.cat((y7,x8),dim=1)
        x8 = self.Conv8(x8)

        x9 = self.Maxpool(x8)
        x9 = self.Conv9(x9)



        

        # decoding + concat path
        d9 = self.Up9(x9)
        x8 = self.Att9(g=d9, x=x8)
        d91=self.aux7(d9)
        d9 = torch.cat((x8,d9),dim=1)
        d9 = self.Up_conv9(d9)

        d8 = self.Up8(d9)
        x7 = self.Att8(g=d8, x=x7)
        d81=self.aux6(d8)
        d8 = torch.cat((x7,d8),dim=1)
        d8 = self.Up_conv8(d8)


        
        d7 = self.Up7(d8)
        x6 = self.Att7(g=d7, x=x6)
        d71=self.aux5(d7)
        d7 = torch.cat((x6,d7),dim=1)
        d7 = self.Up_conv7(d7)

        d6 = self.Up6(d7)
        x5 = self.Att6(g=d6, x=x5)
        d61=self.aux4(d6)
        d6 = torch.cat((x5,d6),dim=1)
        d6 = self.Up_conv6(d6)




        d5 = self.Up5(d6)
        x4 = self.Att5(g=d5, x=x4)
        d51=self.aux3(d5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d41=self.aux2(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d31=self.aux1(d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        #d1 = self.Conv_1x1(d2)
        d1=self.conv(d2)

        d=0.02*d91+0.04*d81+0.06*d71+0.08*d61+0.1*d51+0.2*d41+0.5*d31+1.0*d1


        d = self.output_layer(d)

        return d
"""
class U_Net1(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net1,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.aux3=nn.Sequential(
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            
            nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.aux2 = nn.Sequential(

            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            
            nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.aux1 = nn.Sequential(

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)


        )
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.conv=nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        #self.Conv_1x1 = nn.Conv2d(8,output_ch,kernel_size=1,stride=1,padding=0)
        self.output_layer=nn.Sigmoid()



    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d51=self.aux3(d5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d41=self.aux2(d4)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d31=self.aux1(d3)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1=self.conv(d2)

        #d=0.02*d51+0.08*d41+0.1*d31+0.8*d1


        d1 = self.output_layer(d1)

        return d1
"""