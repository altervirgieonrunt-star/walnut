# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Enhanced Copy-Paste Data Augmentation for Egg Segmentation
增强版复制粘贴数据增强 - 专为鸭蛋分割任务设计

核心功能：
1. 建立实例池（Instance Pool）- 存储所有可复制的egg和barrier实例
2. 随机复制粘贴 - 从实例池中随机选择实例粘贴到目标图像
3. 智能掩码处理 - 处理遮挡关系，更新或删除被遮挡的掩码
4. 边界羽化 - 可选的边界平滑处理

参考论文：
Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
https://arxiv.org/abs/2012.07177
"""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Dict, Tuple

import cv2
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances


class InstancePool:
    """
    实例池 - 存储所有可用于Copy-Paste的目标实例
    
    每个实例包含：
    - image: 抠出的目标图像（带透明通道或原始RGB）
    - mask: 二值化掩码
    - class_id: 类别ID（0=egg, 1=barrier等）
    - bbox: 边界框 [x1, y1, x2, y2]
    """
    
    def __init__(self, max_size: int = 1000):
        """
        初始化实例池
        
        Args:
            max_size: 实例池最大容量（避免内存溢出）
        """
        self.instances: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.egg_instances = []  # 专门存储egg实例
        self.barrier_instances = []  # 专门存储barrier实例
        
    def add_instance(self, image: np.ndarray, mask: np.ndarray, class_id: int, bbox: np.ndarray = None):
        """
        添加一个实例到池中
        
        Args:
            image: 原始图像
            mask: 实例掩码（二值化或多边形格式）
            class_id: 类别ID
            bbox: 边界框（可选，如果没有则从mask计算）
        """
        if len(self.instances) >= self.max_size:
            # 随机删除一个旧实例（FIFO策略）
            self.instances.pop(0)
            
        # 如果mask是多边形格式，转换为二值掩码
        if mask.dtype != bool and mask.max() <= 1:
            mask = mask.astype(bool)
        elif mask.dtype != bool:
            mask = mask > 0
            
        # 计算边界框
        if bbox is None:
            bbox = self._mask_to_bbox(mask)
            
        # 裁剪出实例区域
        x1, y1, x2, y2 = bbox.astype(int)
        if x2 <= x1 or y2 <= y1:
            return  # 无效bbox
            
        # 确保边界在图像范围内
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        cropped_img = image[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2].copy()
        
        instance = {
            'image': cropped_img,
            'mask': cropped_mask,
            'class_id': class_id,
            'bbox': bbox,
            'original_size': (y2 - y1, x2 - x1)  # H, W
        }
        
        self.instances.append(instance)
        
        # 同时添加到分类列表
        if class_id == 0:  # egg
            self.egg_instances.append(instance)
        else:  # barrier
            self.barrier_instances.append(instance)
    
    def get_random_instance(self, prefer_class: int = None, balance_classes: bool = True) -> Dict[str, Any]:
        """
        随机获取一个实例（支持类别平衡采样）
        
        Args:
            prefer_class: 优先选择的类别（None表示随机）
            balance_classes: 是否平衡类别采样（优先采样少数类）
            
        Returns:
            实例字典的深拷贝
        """
        if len(self.instances) == 0:
            return None
        
        # 类别平衡策略：优先采样少数类
        if balance_classes and prefer_class is None:
            n_egg = len(self.egg_instances)
            n_barrier = len(self.barrier_instances)
            
            if n_egg > 0 and n_barrier > 0:
                # 计算采样权重：少数类权重更高
                total = n_egg + n_barrier
                # 如果barrier很少，给它更高的采样概率
                barrier_prob = max(0.3, 1.0 - n_barrier / total)  # 至少30%概率采样barrier
                
                if random.random() < barrier_prob and n_barrier > 0:
                    prefer_class = 1  # barrier
                else:
                    prefer_class = 0  # egg
            elif n_barrier > 0:
                prefer_class = 1
            else:
                prefer_class = 0
        
        # 根据prefer_class选择池
        if prefer_class is not None:
            pool = self.egg_instances if prefer_class == 0 else self.barrier_instances
            if len(pool) == 0:
                pool = self.instances
        else:
            pool = self.instances
            
        return deepcopy(random.choice(pool))
    
    def build_from_dataset(self, dataset, max_images: int = 200, verbose: bool = True, sample_strategy: str = 'uniform'):
        """
        从数据集构建实例池
        
        Args:
            dataset: YOLO数据集对象
            max_images: 最多使用多少张图像构建池
            verbose: 是否打印进度
            sample_strategy: 采样策略 ('sequential'=顺序, 'uniform'=均匀分布, 'random'=随机)
        """
        total_images = len(dataset)
        n_images = min(total_images, max_images)
        n_instances = 0
        
        # 选择图像索引
        if sample_strategy == 'uniform':
            # 均匀分布采样（确保覆盖整个数据集）
            indices = np.linspace(0, total_images-1, n_images, dtype=int)
        elif sample_strategy == 'random':
            # 随机采样
            indices = np.random.choice(total_images, n_images, replace=False)
        else:  # sequential
            indices = range(n_images)
        
        if verbose:
            LOGGER.info(f"Building instance pool from {n_images} images (strategy={sample_strategy})...")
        
        for i in indices:
            try:
                # 获取图像和标签
                data = dataset.get_image_and_label(i)
                img = data['img']
                instances = data.get('instances', None)
                
                if instances is None or len(instances) == 0:
                    continue
                    
                # 提取每个实例
                h, w = img.shape[:2]
                
                # 确保instances有segments
                if not hasattr(instances, 'segments') or len(instances.segments) == 0:
                    continue
                
                segments = instances.segments
                
                # 正确提取类别信息 - 类别在data['cls']中，而不是instances.cls
                cls_data = data.get('cls', None)
                if cls_data is not None:
                    # 处理不同的cls格式
                    if hasattr(cls_data, 'cpu'):  # torch tensor
                        cls_data = cls_data.cpu().numpy()
                    if hasattr(cls_data, 'flatten'):  # numpy array
                        cls_data = cls_data.flatten()
                    classes = cls_data
                else:
                    # 如果data中也没有cls，尝试从instances中获取
                    if hasattr(instances, 'cls'):
                        cls_data = instances.cls
                        if hasattr(cls_data, 'cpu'):
                            cls_data = cls_data.cpu().numpy()
                        if hasattr(cls_data, 'flatten'):
                            cls_data = cls_data.flatten()
                        classes = cls_data
                    else:
                        classes = np.zeros(len(segments))
                
                for j, seg in enumerate(segments):
                    # 将segment转换为mask
                    # 注意：segments是归一化坐标(0-1)，需要转换为像素坐标
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    # 转换归一化坐标到像素坐标
                    seg_pixels = seg.copy()
                    seg_pixels[:, 0] *= w  # x坐标
                    seg_pixels[:, 1] *= h  # y坐标
                    seg_int = seg_pixels.astype(np.int32)
                    
                    if len(seg_int) > 0:
                        cv2.fillPoly(mask, [seg_int], 1)
                        
                        # 添加到实例池
                        class_id = int(classes[j]) if j < len(classes) else 0
                        self.add_instance(img, mask.astype(bool), class_id)
                        n_instances += 1
                        
                        # 调试：记录barrier实例
                        if class_id == 1 and verbose and len(self.barrier_instances) <= 5:
                            LOGGER.info(f"  Found barrier in image {i}: class_id={class_id}")
                        
            except Exception as e:
                if verbose:
                    LOGGER.warning(f"Error processing image {i}: {e}")
                continue
        
        if verbose:
            n_egg = len(self.egg_instances)
            n_barrier = len(self.barrier_instances)
            LOGGER.info(f"Instance pool built: {n_instances} instances total")
            LOGGER.info(f"  - Eggs: {n_egg} ({100*n_egg/max(n_instances,1):.1f}%)")
            LOGGER.info(f"  - Barriers: {n_barrier} ({100*n_barrier/max(n_instances,1):.1f}%)")
            
            # 类别不平衡警告
            if n_barrier == 0 and n_egg > 0:
                LOGGER.warning("⚠️  No barrier instances found! Only egg instances will be pasted.")
                LOGGER.warning("   Consider increasing cp_pool_images to find more barrier examples.")
            elif n_barrier > 0 and n_egg / max(n_barrier, 1) > 10:
                LOGGER.warning(f"⚠️  Class imbalance detected: {n_egg/max(n_barrier,1):.1f}:1 (egg:barrier)")
                LOGGER.warning("   Using balanced sampling to prefer minority class (barrier).")
    
    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> np.ndarray:
        """从mask计算边界框"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return np.array([0, 0, 1, 1])
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return np.array([x1, y1, x2 + 1, y2 + 1])
    
    def __len__(self):
        return len(self.instances)


class CopyPasteEnhanced:
    """
    增强版Copy-Paste数据增强
    
    特点：
    1. 使用实例池存储可复制的实例
    2. 智能处理掩码遮挡关系
    3. 删除完全被遮挡的掩码
    4. 更新部分被遮挡的掩码
    5. 可选的边界羽化效果
    """
    
    def __init__(
        self,
        instance_pool: InstancePool = None,
        p: float = 0.5,
        n_paste: Tuple[int, int] = (1, 5),
        scale_range: Tuple[float, float] = (0.5, 1.5),
        min_area_threshold: int = 100,
        blend_edges: bool = False,
        blend_kernel: int = 5,
        balance_classes: bool = True  # 新增：是否平衡类别
    ):
        """
        初始化Copy-Paste增强器
        
        Args:
            instance_pool: 实例池对象
            p: 应用增强的概率
            n_paste: 每次粘贴的实例数量范围 (min, max)
            scale_range: 粘贴实例的缩放范围
            min_area_threshold: 最小面积阈值（小于此值的mask会被删除）
            blend_edges: 是否对边缘进行羽化
            blend_kernel: 羽化核大小
        """
        self.instance_pool = instance_pool if instance_pool is not None else InstancePool()
        self.p = p
        self.n_paste = n_paste
        self.scale_range = scale_range
        self.min_area_threshold = min_area_threshold
        self.blend_edges = blend_edges
        self.blend_kernel = blend_kernel
        self.balance_classes = balance_classes
        
    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        对标签数据应用Copy-Paste增强
        
        Args:
            labels: YOLO标签字典，包含 'img', 'instances' 等
            
        Returns:
            增强后的标签字典
        """
        # 检查是否应用增强
        if random.random() > self.p:
            return labels
        
        # 检查实例池是否为空
        if len(self.instance_pool) == 0:
            return labels
        
        # 检查是否有segments
        if 'instances' not in labels or len(labels['instances']) == 0:
            return labels
            
        instances = labels['instances']
        if not hasattr(instances, 'segments') or len(instances.segments) == 0:
            return labels
        
        # 执行Copy-Paste
        return self._copy_paste_transform(labels)
    
    def _copy_paste_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        执行Copy-Paste变换的核心逻辑
        """
        img = labels['img'].copy()
        h, w = img.shape[:2]
        instances = labels['instances']
        
        # 将当前实例转换为mask列表 + 保存对应的类别（一对一）
        current_masks = self._instances_to_masks(instances, h, w)
        
        # 🔧 关键修复：cls在labels['cls']中，不在instances中！
        # 必须从labels中获取，而不是从instances中
        if 'cls' in labels:
            current_classes = labels['cls']
            # 处理不同的数据类型
            if hasattr(current_classes, 'cpu'):
                current_classes = current_classes.cpu().numpy()
            if hasattr(current_classes, 'flatten'):
                current_classes = current_classes.flatten()
        else:
            # 如果labels中也没有cls，尝试从instances获取（向后兼容）
            if hasattr(instances, 'cls'):
                current_classes = instances.cls.cpu().numpy() if hasattr(instances.cls, 'cpu') else instances.cls
            else:
                # 最后的fallback：设为0（但这种情况不应该发生）
                current_classes = np.zeros(len(instances))
        
        # ⚠️ 关键：将masks和classes成对存储，确保同步删除
        current_items = list(zip(current_masks, current_classes))
        
        # 确定要粘贴的实例数量
        n_to_paste = random.randint(self.n_paste[0], self.n_paste[1])
        
        # 存储新增的items (mask, class_id)
        new_items = []
        
        for _ in range(n_to_paste):
            # 从实例池中随机选择一个实例（使用类别平衡策略）
            source_instance = self.instance_pool.get_random_instance(balance_classes=self.balance_classes)
            if source_instance is None:
                continue
            
            # 根据类别选择不同的缩放策略
            # egg(class 0): 不缩放或轻微缩放（本身很小）
            # barrier(class 1): 可以大幅缩放（本身很大）
            class_id = source_instance['class_id']
            if class_id == 0:  # egg
                scale_range_for_class = (1.0, 1.0)  # 不缩放
            elif class_id == 1:  # barrier
                scale_range_for_class = (0.5, 1.5)  # 大幅缩放（可以缩小）
            else:
                scale_range_for_class = self.scale_range  # 其他类别使用默认
            
            # 应用随机变换（缩放、旋转等）
            transformed = self._transform_instance(source_instance, scale_range_for_class)
            
            # 选择粘贴位置
            paste_x, paste_y = self._get_paste_position(transformed, h, w)
            
            # 创建粘贴mask（在目标图像坐标系中）
            paste_mask = np.zeros((h, w), dtype=bool)
            inst_h, inst_w = transformed['mask'].shape
            
            # 确保不超出边界
            y_end = min(paste_y + inst_h, h)
            x_end = min(paste_x + inst_w, w)
            inst_h_actual = y_end - paste_y
            inst_w_actual = x_end - paste_x
            
            if inst_h_actual <= 0 or inst_w_actual <= 0:
                continue
                
            paste_mask[paste_y:y_end, paste_x:x_end] = transformed['mask'][:inst_h_actual, :inst_w_actual]
            
            # 粘贴图像内容
            if self.blend_edges:
                # 边缘羽化
                img = self._blend_paste(img, transformed['image'], paste_mask, paste_x, paste_y)
            else:
                # 直接替换
                img[paste_mask] = transformed['image'][:inst_h_actual, :inst_w_actual][
                    transformed['mask'][:inst_h_actual, :inst_w_actual]
                ]
            
            # 🔧 修复BUG：处理遮挡时同步更新masks和classes
            current_items = self._handle_occlusion_with_classes(current_items, paste_mask)
            
            # 添加新item (mask, class_id)
            new_items.append((paste_mask, transformed['class_id']))
        
        # 合并current和new items
        all_items = current_items + new_items
        
        # 分离masks和classes
        all_masks = [item[0] for item in all_items]
        all_classes = np.array([item[1] for item in all_items]) if all_items else np.array([])
        
        # 过滤掉面积太小的masks
        valid_indices = []
        for i, mask in enumerate(all_masks):
            if mask.sum() >= self.min_area_threshold:
                valid_indices.append(i)
        
        all_masks = [all_masks[i] for i in valid_indices]
        all_classes = all_classes[valid_indices] if len(valid_indices) > 0 else np.array([])
        
        # 更新labels
        labels['img'] = img
        
        if len(all_masks) > 0:
            # 将masks转换回instances格式
            new_instances = self._masks_to_instances(all_masks, all_classes, h, w)
            labels['instances'] = new_instances
            # 🔧 关键修复：使用排序后的cls（从instances中获取），而不是未排序的all_classes
            labels['cls'] = new_instances.cls.reshape(-1, 1)
        else:
            # 没有有效实例 - 必须返回正确格式的空numpy数组
            labels['instances'] = Instances(
                bboxes=np.empty((0, 4)),
                segments=np.zeros((0, 1000, 2), dtype=np.float32),  # 空的3D数组，不是列表
                bbox_format='xyxy',
                normalized=False
            )
            labels['cls'] = np.empty((0, 1))
        
        return labels
    
    def _instances_to_masks(self, instances, h: int, w: int) -> List[np.ndarray]:
        """将Instances对象转换为mask列表"""
        masks = []
        segments = instances.segments
        
        for seg in segments:
            mask = np.zeros((h, w), dtype=np.uint8)
            seg_int = seg.astype(np.int32)
            if len(seg_int) > 0:
                cv2.fillPoly(mask, [seg_int], 1)
                masks.append(mask.astype(bool))
        
        return masks
    
    def _masks_to_instances(self, masks: List[np.ndarray], classes: np.ndarray, h: int, w: int) -> Instances:
        """
        将mask列表转换回Instances对象
        
        🔧 关键修复：按mask面积排序，与Format._format_segments保持一致！
        这样Format在重新排序时，顺序不会改变，cls和instances就能正确对应
        """
        from ultralytics.utils.ops import resample_segments
        
        segments = []
        bboxes = []
        areas = []  # 🔧 新增：记录每个mask的面积
        
        for mask in masks:
            # 提取轮廓
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) > 0:
                # 使用最大的轮廓
                contour = max(contours, key=cv2.contourArea)
                segment = contour.reshape(-1, 2).astype(np.float32)
                segments.append(segment)
                
                # 计算bbox
                x1, y1 = segment.min(axis=0)
                x2, y2 = segment.max(axis=0)
                bboxes.append([x1, y1, x2, y2])
                
                # 🔧 计算面积（与polygons2masks_overlap保持一致）
                areas.append(mask.sum())
        
        if len(segments) == 0:
            # 返回空的Instances，格式与dataset.py保持一致
            return Instances(
                bboxes=np.empty((0, 4)),
                segments=np.zeros((0, 1000, 2), dtype=np.float32),
                bbox_format='xyxy',
                normalized=False
            )
        
        # 🔧 关键修复：按面积从大到小排序（与Format._format_segments一致）
        areas = np.array(areas)
        sorted_idx = np.argsort(-areas)  # 从大到小
        
        segments = [segments[i] for i in sorted_idx]
        bboxes = [bboxes[i] for i in sorted_idx]
        classes = classes[sorted_idx]  # 同步排序classes
        
        # Resample segments to 1000 points (与dataset.py保持一致)
        segments_resampled = resample_segments(segments, n=1000)
        segments_array = np.stack(segments_resampled, axis=0)  # Shape: (N, 1000, 2)
        
        instances = Instances(
            bboxes=np.array(bboxes),
            segments=segments_array,
            bbox_format='xyxy',
            normalized=False  # 我们的坐标是像素坐标，未归一化
        )
        instances.cls = classes
        
        return instances
    
    def _transform_instance(self, instance: Dict[str, Any], scale_range: Tuple[float, float]) -> Dict[str, Any]:
        """
        对实例应用随机变换（缩放、旋转等）
        
        Args:
            instance: 实例字典
            scale_range: 缩放范围
            
        Returns:
            变换后的实例
        """
        img = instance['image'].copy()
        mask = instance['mask'].copy()
        
        # 随机缩放
        scale = random.uniform(scale_range[0], scale_range[1])
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        if new_h > 0 and new_w > 0:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # 可以添加更多变换：旋转、翻转等
        # 随机水平翻转
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask.astype(np.uint8), 1).astype(bool)
        
        return {
            'image': img,
            'mask': mask,
            'class_id': instance['class_id']
        }
    
    def _get_paste_position(self, instance: Dict[str, Any], img_h: int, img_w: int) -> Tuple[int, int]:
        """
        获取粘贴位置（随机）
        
        Args:
            instance: 实例字典
            img_h: 目标图像高度
            img_w: 目标图像宽度
            
        Returns:
            (x, y) 左上角坐标
        """
        inst_h, inst_w = instance['mask'].shape
        
        # 确保粘贴后实例完全在图像内
        max_x = max(0, img_w - inst_w)
        max_y = max(0, img_h - inst_h)
        
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        return x, y
    
    def _handle_occlusion(self, existing_masks: List[np.ndarray], new_mask: np.ndarray) -> List[np.ndarray]:
        """
        处理新粘贴mask对现有masks的遮挡
        
        核心逻辑：
        1. 对每个现有mask，计算与new_mask的交集
        2. 如果有交集，从现有mask中减去交集部分
        3. 如果减去后面积太小，删除该mask
        
        Args:
            existing_masks: 现有mask列表
            new_mask: 新粘贴的mask
            
        Returns:
            更新后的mask列表
        """
        updated_masks = []
        
        for mask in existing_masks:
            # 计算交集
            intersection = mask & new_mask
            
            if intersection.sum() > 0:
                # 有遮挡，更新mask
                updated_mask = mask & ~new_mask
                
                # 检查剩余面积
                if updated_mask.sum() >= self.min_area_threshold:
                    updated_masks.append(updated_mask)
                # 否则该mask被完全遮挡，丢弃
            else:
                # 无遮挡，保留原mask
                updated_masks.append(mask)
        
        return updated_masks
    
    def _handle_occlusion_with_classes(self, existing_items: List[Tuple[np.ndarray, int]], new_mask: np.ndarray) -> List[Tuple[np.ndarray, int]]:
        """
        处理新粘贴mask对现有items的遮挡，同时保持mask和class_id的同步
        
        Args:
            existing_items: 现有items列表，每个item是(mask, class_id)的元组
            new_mask: 新粘贴的mask
            
        Returns:
            更新后的items列表，保证mask和class_id一一对应
        """
        updated_items = []
        
        for mask, class_id in existing_items:
            # 计算交集
            intersection = mask & new_mask
            
            if intersection.sum() > 0:
                # 有遮挡，更新mask
                updated_mask = mask & ~new_mask
                
                # 检查剩余面积
                if updated_mask.sum() >= self.min_area_threshold:
                    updated_items.append((updated_mask, class_id))  # 保持class_id不变
                # 否则该mask被完全遮挡，丢弃（class_id也一起丢弃）
            else:
                # 无遮挡，保留原item
                updated_items.append((mask, class_id))
        
        return updated_items
    
    def _blend_paste(
        self,
        target_img: np.ndarray,
        source_img: np.ndarray,
        paste_mask: np.ndarray,
        paste_x: int,
        paste_y: int
    ) -> np.ndarray:
        """
        带边缘羽化的粘贴
        
        Args:
            target_img: 目标图像
            source_img: 源实例图像
            paste_mask: 粘贴mask
            paste_x, paste_y: 粘贴位置
            
        Returns:
            混合后的图像
        """
        # 提取粘贴区域的mask
        h, w = target_img.shape[:2]
        inst_h, inst_w = source_img.shape[:2]
        
        y_end = min(paste_y + inst_h, h)
        x_end = min(paste_x + inst_w, w)
        
        # 对mask边缘进行高斯模糊
        mask_region = paste_mask[paste_y:y_end, paste_x:x_end].astype(np.uint8)
        blurred_mask = cv2.GaussianBlur(mask_region * 255, (self.blend_kernel, self.blend_kernel), 0) / 255.0
        
        # Alpha混合
        inst_h_actual = y_end - paste_y
        inst_w_actual = x_end - paste_x
        
        for c in range(3):
            target_img[paste_y:y_end, paste_x:x_end, c] = (
                blurred_mask * source_img[:inst_h_actual, :inst_w_actual, c] +
                (1 - blurred_mask) * target_img[paste_y:y_end, paste_x:x_end, c]
            )
        
        return target_img


def build_instance_pool_from_dataset(dataset, max_images: int = 200, max_pool_size: int = 1000, sample_strategy: str = 'uniform') -> InstancePool:
    """
    便捷函数：从数据集构建实例池
    
    Args:
        dataset: YOLO数据集对象
        max_images: 最多使用多少张图像
        max_pool_size: 实例池最大容量
        sample_strategy: 采样策略 ('uniform'=均匀, 'random'=随机, 'sequential'=顺序)
        
    Returns:
        构建好的实例池
    """
    pool = InstancePool(max_size=max_pool_size)
    pool.build_from_dataset(dataset, max_images=max_images, verbose=True, sample_strategy=sample_strategy)
    return pool

