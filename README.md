>## Garbage_Classify-2019华为云垃圾分类大赛

### 框架
- tensorflow, keras
### 尝试的网络
- NasNetLarge
- SeResNext101
- 模型融合
### 图像数据增强
- 旋转, 水平翻转, 裁剪(by imgaug)
- mixup-augmentation
### 训练技巧
- label_smoothing
- kernel_regularizer, activity_regularizer
- EarlyStop
- reduce_lr (ReduceLROnPlateau)

### 测试增强(TTA)
- multi-crop


 
