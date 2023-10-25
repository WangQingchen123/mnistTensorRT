import torch
from torch import nn
import torchvision
import os
import struct
from torchsummary import summary

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('F:\\jetson\\mnistTorch\\model.pth')
    net = net.to('cuda:0')
    net.eval()
    print('model:',net)
    tmp=torch.ones(1,1,28,28).to('cuda:0')
    print('input:', tmp)
    out = net(tmp)

    print('output:',out)

    summary(net,(1,28,28))
    f = open("ministCNN.wts",'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key:',k)
        print('value:',v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k,len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f",float(vv)).hex())
        f.write("\n")


if __name__ == '__main__':
    main()