import torch
import torch.nn as nn

import effdet
from effdet import EfficientDet


from models.fusion_modules import CBAMLayer, attention_block, shuffle_attention_block, MRF,SKNet

##################################### Attention Fusion Net ###############################################
class Att_FusionNet(nn.Module):

    def __init__(self, args):
        super(Att_FusionNet, self).__init__()

        self.config = effdet.config.model_config.get_efficientdet_config('efficientdetv2_dt')
        self.config.num_classes = args.num_classes
        if "stf" in args.dataset:
            self.config.image_size = (1280, 1280)

        thermal_det = EfficientDet(self.config)
        rgb_det = EfficientDet(self.config)

        if args.thermal_checkpoint_path:
            effdet.helpers.load_checkpoint(thermal_det, args.thermal_checkpoint_path)
            print('Loading Thermal from {}'.format(args.thermal_checkpoint_path))
        else:
            print('Thermal checkpoint path not provided.')
        
        if args.rgb_checkpoint_path:
            effdet.helpers.load_checkpoint(rgb_det, args.rgb_checkpoint_path)
            print('Loading RGB from {}'.format(args.rgb_checkpoint_path))
        else:
            if 'flir' in args.dataset:
                effdet.helpers.load_pretrained(rgb_det, self.config.url)
                print('Loading RGB from {}'.format(self.config.url))
            print('RGB checkpoint path not provided.')
            

            
        
        self.thermal_backbone = thermal_det.backbone
        self.thermal_fpn = thermal_det.fpn
        self.thermal_class_net = thermal_det.class_net
        self.thermal_box_net = thermal_det.box_net

        self.rgb_backbone = rgb_det.backbone
        self.rgb_fpn = rgb_det.fpn
        self.rgb_class_net = rgb_det.class_net
        self.rgb_box_net = rgb_det.box_net

        fusion_det = EfficientDet(self.config)
        
        if args.init_fusion_head_weights == 'thermal':
            effdet.helpers.load_checkpoint(fusion_det, args.thermal_checkpoint_path) # This is optional
            print("Loading fusion head from thermal checkpoint.")
        elif args.init_fusion_head_weights == 'rgb':
            effdet.helpers.load_checkpoint(fusion_det, args.rgb_checkpoint_path)
            print("Loading fusion head from rgb checkpoint.")
        else:
            print('Fusion head random init.')
        

        self.fusion_class_net = fusion_det.class_net
        self.fusion_box_net = fusion_det.box_net

        if args.branch == 'fusion':
            self.attention_type = args.att_type
            in_chs = args.channels
            for level in range(self.config.num_levels):
                if self.attention_type=="cbam":
                    self.add_module("fusion_"+self.attention_type+str(level), CBAMLayer(2*in_chs))
                elif self.attention_type=="eca":
                    self.add_module("fusion_"+self.attention_type+str(level), attention_block(2*in_chs))
                elif self.attention_type=="shuffle":
                    self.add_module("fusion_"+self.attention_type+str(level), shuffle_attention_block(2*in_chs))
                elif self.attention_type=="mrf":
                    self.add_module("fusion_"+self.attention_type+str(level), MRF())
                elif self.attention_type == "sknet":
                    self.add_module("fusion_" + self.attention_type + str(level), SKNet())
                else:
                    raise ValueError('Attention type not supported.')

    def forward(self, data_pair, branch='fusion'):
        thermal_x, rgb_x = data_pair[0], data_pair[1]

        class_net = getattr(self, f'{branch}_class_net')
        box_net = getattr(self, f'{branch}_box_net')
        
        x = None
        if branch =='fusion':
            thermal_x = self.thermal_backbone(thermal_x)
            rgb_x = self.rgb_backbone(rgb_x)

            thermal_x = self.thermal_fpn(thermal_x)
            rgb_x = self.rgb_fpn(rgb_x)

            out = []
            if self.attention_type=="mrf" or self.attention_type=="sknet":
                for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                    attention = getattr(self, "fusion_" + self.attention_type + str(i))
                    out.append(attention(tx,vx))

            else:
                for i, (tx, vx) in enumerate(zip(thermal_x, rgb_x)):
                    x = torch.cat((tx, vx), dim=1)
                    attention = getattr(self, "fusion_" + self.attention_type + str(i))
                    out.append(attention(x))
        else:
            fpn = getattr(self, f'{branch}_fpn')
            backbone = getattr(self, f'{branch}_backbone')
            if branch =='thermal':
                x = thermal_x
            elif branch =='rgb':
                x = rgb_x
            feats = backbone(x)
            out = fpn(feats)
        
        
        x_class = class_net(out)
        x_box = box_net(out)

        return x_class, x_box


