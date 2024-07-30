import torch
import numpy as np
import math
from torch import nn
from model.stylegan.model import Generator

class VToonifyResBlock(nn.Module):
    def __init__(self, fin):
        super().__init__()

        self.conv = nn.Conv2d(fin, fin, 3,  1, 1)
        self.conv2 = nn.Conv2d(fin, fin, 3,  1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        out = self.lrelu(self.conv(x))
        out = self.lrelu(self.conv2(out))      
        out = (out + x) / math.sqrt(2)
        return out    

class VToonify(nn.Module):
    def __init__(self,
                 in_size=256,
                 out_size=1024,
                 img_channels=3,
                 style_channels=512,
                 num_mlps=8,
                 channel_multiplier=2,
                 num_res_layers=6
                ):

        super().__init__()

        # StyleGANv2, with weights being fixed
        self.generator = Generator(out_size, style_channels, num_mlps, channel_multiplier)

        self.in_size = in_size
        self.style_channels = style_channels
        channels = self.generator.channels

        # encoder
        num_styles = int(np.log2(out_size)) * 2 - 2
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 4, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(img_channels+19, 32, 3, 1, 1, bias=True), 
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(32, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        
        for res in encoder_res:
            in_channels = channels[res]
            if res > 32:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
                self.encoder.append(block)
            else:
                layers = []
                for _ in range(num_res_layers):
                    layers.append(VToonifyResBlock(in_channels))
                self.encoder.append(nn.Sequential(*layers))
                block = nn.Conv2d(in_channels, img_channels, 1, 1, 0, bias=True)
                self.encoder.append(block)
        
        # trainable fusion module
        self.fusion_out = nn.ModuleList()
        self.fusion_skip = nn.ModuleList()
        for res in encoder_res[::-1]:
            num_channels = channels[res]
            self.fusion_out.append(
                nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=True))

            self.fusion_skip.append(
                nn.Conv2d(num_channels + 3, 3, 3, 1, 1, bias=True))
    
    def forward(self, x, style, d_s=None, return_mask=False, return_feat=False):
        # map style to W+ space
        if style is not None and style.ndim < 3:
            adastyles = style.unsqueeze(1).repeat(1, self.generator.n_latent, 1)
        elif style is not None:
            nB, nL, nD = style.shape
            adastyles = style

        # obtain multi-scale content features
        feat = x
        encoder_features = []
        # downsampling conv parts of E
        for block in self.encoder[:-2]:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]
        # Resblocks in E
        for ii, block in enumerate(self.encoder[-2]):
            feat = block(feat)
        # the last-layer feature of E (inputs of backbone)
        out = feat
        skip = self.encoder[-1](feat)
        if return_feat:
            return out, skip
        
        # 32x32 ---> higher res
        _index = 1
        m_Es = []
        for conv1, conv2, to_rgb in zip(
            self.stylegan().convs[6::2], self.stylegan().convs[7::2], self.stylegan().to_rgbs[3:]): 
            
            # pass the mid-layer features of E to the corresponding resolution layers of G
            if 2 ** (5+((_index-1)//2)) <= self.in_size:
                fusion_index = (_index - 1) // 2
                f_E = encoder_features[fusion_index]

                out = self.fusion_out[fusion_index](torch.cat([out, f_E], dim=1))
                skip = self.fusion_skip[fusion_index](torch.cat([skip, f_E], dim=1))  
            
            # remove the noise input
            batch, _, height, width = out.shape
            noise = x.new_empty(batch, 1, height * 2, width * 2).normal_().detach() * 0.0
            
            out = conv1(out, adastyles[:, _index+6], noise=noise)
            out = conv2(out, adastyles[:, _index+7], noise=noise)
            skip = to_rgb(out, adastyles[:, _index+8], skip)
            _index += 2

        image = skip
        return image
    
    def stylegan(self):
        return self.generator
        
    def zplus2wplus(self, zplus):
        return self.stylegan().style(zplus.reshape(zplus.shape[0]*zplus.shape[1], zplus.shape[2])).reshape(zplus.shape)