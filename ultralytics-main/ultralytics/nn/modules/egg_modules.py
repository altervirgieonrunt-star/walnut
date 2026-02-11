import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, autopad, h_sigmoid, h_swish, CoordAtt
from .block import Bottleneck

__all__ = (
    "ColorShapeAttention",
    "ColorAttention", 
    "DCNv2_Pure",
    "EllipseConv",
    "C2f_DCN",
    "Bottleneck_DCN",
    "C2f_CSA",
)


class ColorAttention(nn.Module):
    """
    颜色注意力模块
    """
    
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels (int)
            reduction (int)
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out


class ColorShapeAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels (int)
            reduction (int)
        """
        super().__init__()
        # 颜色注意力分支
        self.color_attn = ColorAttention(channels, reduction)
        # 形状注意力分支（使用Coordinate Attention）
        self.shape_attn = CoordAtt(channels, channels, reduction)
        
        # 融合权重（可学习）
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor):
            
        Returns:
            torch.Tensor:
        """
        color_out = self.color_attn(x)
        shape_out = self.shape_attn(x)
        out = self.alpha * color_out + self.beta * shape_out
        return out


class DCNv2_Pure(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, deformable_groups=1):
        """
        Args:
            in_channels (int)
            out_channels (int)
            kernel_size (int)
            stride (int)
            padding (int)
            dilation (int)
            deformable_groups (int)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride if isinstance(stride, int) else stride[0]
        self.padding = padding if isinstance(padding, int) else padding[0]
        self.dilation = dilation if isinstance(dilation, int) else dilation[0]
        self.deformable_groups = deformable_groups
        
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * deformable_groups * self.kernel_size * self.kernel_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )
        
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)
        
    def forward(self, x):
        """
        
        Args:
            x (torch.Tensor):
            
        Returns:
            torch.Tensor:
        """
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat([o1, o2], dim=1)
        mask = torch.sigmoid(mask)
        
        return self._deform_conv(x, offset, mask)
    
    def _deform_conv(self, x, offset, mask):
        """
        
        Args:
            x (torch.Tensor):
            offset (torch.Tensor)  [B, 2*K*K, H, W]
            mask (torch.Tensor) [B, K*K, H, W]
            
        Returns:
            torch.Tensor [B, C_out, H, W]
        """
        B, C, H, W = x.shape
        K = self.kernel_size
        
        grid_h, grid_w = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        
        kernel_h, kernel_w = torch.meshgrid(
            torch.arange(-(K//2), K//2 + 1, device=x.device, dtype=x.dtype),
            torch.arange(-(K//2), K//2 + 1, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        kernel_offsets = torch.stack([kernel_w.flatten(), kernel_h.flatten()], dim=1)  
        
        offset = offset.view(B, 2, K*K, H, W).permute(0, 2, 3, 4, 1)  
        
        base_grid = torch.stack([grid_w, grid_h], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).unsqueeze(1)  # [1, 1, H, W, 2]
        kernel_offsets = kernel_offsets.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, K*K, 1, 1, 2]
        
        # 采样坐标 = 基础网格 + kernel偏移 + 学习的offset
        sample_grid = base_grid + kernel_offsets + offset  # [B, K*K, H, W, 2]
        
        # 归一化到[-1, 1]
        norm_grid = torch.zeros_like(sample_grid)
        norm_grid[..., 0] = 2.0 * sample_grid[..., 0] / max(W - 1, 1) - 1.0
        norm_grid[..., 1] = 2.0 * sample_grid[..., 1] / max(H - 1, 1) - 1.0
        
        # Reshape为grid_sample需要的格式 [B*K*K, H, W, 2]
        norm_grid = norm_grid.reshape(B * K * K, H, W, 2)
        
        # 复制input [B, C, H, W] -> [B*K*K, C, H, W]
        x_repeated = x.unsqueeze(1).repeat(1, K*K, 1, 1, 1).reshape(B * K * K, C, H, W)
        
        # 使用grid_sample采样 [B*K*K, C, H, W]
        sampled = F.grid_sample(
            x_repeated, norm_grid,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        # Reshape回 [B, K*K, C, H, W]
        sampled = sampled.reshape(B, K*K, C, H, W)
        
        # 应用mask [B, K*K, H, W] -> [B, K*K, 1, H, W]
        sampled = sampled * mask.unsqueeze(2)
        
        # Reshape为 [B, C, K*K, H, W] 然后展开为 [B, C, K, K, H, W]
        sampled = sampled.permute(0, 2, 1, 3, 4).reshape(B, C, K, K, H, W)
        
        # 应用卷积权重 [C_out, C_in, K, K] * [B, C_in, K, K, H, W]
        # -> [B, C_out, H, W]
        weight = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, C_out, C_in, K, K, 1, 1]
        output = (sampled.unsqueeze(1) * weight).sum(dim=(2, 3, 4))  # [B, C_out, H, W]
        
        # 加bias
        output = output + self.bias.view(1, -1, 1, 1)
        
        return output


class EllipseConv(nn.Module):
    """
    椭圆感知卷积 - 针对圆形/椭圆形目标设计的自适应卷积
    
    核心思想：
    - 学习椭圆形采样模式
    - 沿椭圆边界采样，减少背景信息
    - 对圆形鸭蛋目标有更好的特征提取能力
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            kernel_size (int): 卷积核大小
            stride (int): 步长
            padding (int): 填充
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 标准卷积权重
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        # 椭圆mask（可学习的椭圆形状参数）
        # 初始化为圆形，后续可以学习成椭圆
        self.ellipse_mask = nn.Parameter(self._create_ellipse_mask())
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        
    def _create_ellipse_mask(self):
        """创建初始椭圆mask"""
        K = self.kernel_size
        center = K // 2
        y, x = torch.meshgrid(
            torch.arange(K, dtype=torch.float32),
            torch.arange(K, dtype=torch.float32),
            indexing='ij'
        )
        # 创建圆形mask（后续可学习为椭圆）
        mask = ((x - center)**2 + (y - center)**2) <= (center + 0.5)**2
        return mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入 [B, C, H, W]
            
        Returns:
            torch.Tensor: 输出 [B, C_out, H, W]
        """
        # 应用椭圆mask到卷积权重
        masked_weight = self.weight * torch.sigmoid(self.ellipse_mask)
        
        # 执行卷积
        output = F.conv2d(
            x, masked_weight, self.bias,
            stride=self.stride, padding=self.padding
        )
        return output


class Bottleneck_DCN(nn.Module):
    """使用可变形卷积的Bottleneck块"""
    
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            shortcut (bool): 是否使用shortcut
            g (int): 组卷积数
            k (tuple): 卷积核大小
            e (float): 扩展比例
        """
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        # 使用DCN替代标准卷积
        self.cv2 = DCNv2_Pure(c_, c2, k[1], 1, autopad(k[1], None))
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2
        
    def forward(self, x):
        """前向传播"""
        out = self.act(self.bn(self.cv2(self.cv1(x))))
        return x + out if self.add else out


class C2f_DCN(nn.Module):
    """
    使用可变形卷积的C2f模块
    在Bottleneck中使用DCN，提升对椭圆形目标的适应性
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck数量
            shortcut (bool): 是否使用shortcut
            g (int): 组卷积数
            e (float): 扩展比例
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        # 使用DCN Bottleneck
        self.m = nn.ModuleList(
            Bottleneck_DCN(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        
    def forward(self, x):
        """前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f_CSA(nn.Module):
    """
    使用ColorShapeAttention的C2f模块
    在特征提取后应用双重注意力机制
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): Bottleneck数量
            shortcut (bool): 是否使用shortcut
            g (int): 组卷积数
            e (float): 扩展比例
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        # 添加ColorShapeAttention
        self.csa = ColorShapeAttention(c2, reduction=16)
        
    def forward(self, x):
        """前向传播"""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        # 应用双重注意力
        return self.csa(out)

