论文题目：Multi-scale Wavelet and Attention Enhanced YOLOv11 for UAV-based Forest Fire Detection 

发表会议：2025 7th International Academic Exchange Conference on Science and Technology Innovation (IAECST)

本项目为 MWAE-YOLO 的核心模块实现代码，基于 YOLOv11 框架，面向无人机视角下的森林火灾早期识别任务，重点提升对火焰、烟雾在复杂背景（云雾、强光斑等）下的特征表达能力。

项目开源了论文中提出的三个关键改进模块以及对应的模型配置文件（YAML），用于支持方法复现与工程验证。

<img width="8681" height="6715" alt="MWAE-YOLOv11" src="https://github.com/user-attachments/assets/0ff86def-a254-4ab9-885f-801e2108f00b" />


开源内容说明

本仓库主要包含以下内容：

1. MWAF 模块
   
文件：MWAF.py

功能说明：
引入小波变换对多尺度频域信息进行建模，并与空洞卷积特征进行融合，以增强模型对火焰与烟雾细粒度纹理和结构信息的感知能力。
该模块对应论文中的 MWAF 模块描述部分。

<img width="4442" height="3625" alt="WMAF" src="https://github.com/user-attachments/assets/fb42269a-5bdd-446f-a47b-ff27bb2d2e10" />


2. C2SCSA 模块
   
文件：C2SCSA.py

功能说明：
在特征提取阶段引入跨通道与空间协同注意力机制，提升模型对关键区域的响应能力，抑制复杂背景干扰。
用于增强中高层语义特征的判别性。

其中SCSA建议阅读原文：
https://www.sciencedirect.com/science/article/abs/pii/S0925231225005387

<img width="9475" height="7429" alt="C2SCSA" src="https://github.com/user-attachments/assets/dd9873ba-2808-4a23-b94e-8cf77d852145" />


3. EMA Attention 模块
   
文件：EMA_attention.py

建议阅读原文：
http://dx.doi.org/10.1109/ICASSP49357.2023.10096516

4. 模型结构配置文件（YAML）
   
文件：yolo11n-MWAF-C2SCSA-EMA.yaml

内容说明：
该 YAML 文件定义了集成 MWAF、C2SCSA 与 EMA 模块后的 YOLOv11 网络结构，可直接用于模型构建与训练。

该配置与论文实验中使用的模型结构保持一致。

复现与使用说明

本项目基于 PyTorch 框架实现,模块代码可直接集成至 YOLOv11 的网络定义中,YAML文件可作为模型结构配置参考或直接使用。

数据集地址：https://universe.roboflow.com/yun-anaeo/forest-fire-adk4v

⚠️ 说明：
本仓库主要用于方法复现与模块级验证，完整训练流程、数据集及实验参数请以论文正文描述为准。

开源声明

本项目开源的目的是提升研究工作的透明性与可复现性，仅用于学术研究与非商业用途。如需用于其他用途，请提前联系作者。


联系方式

如在代码使用、复现或理解过程中存在任何问题，欢迎通过以下方式联系作者：

📧 邮箱：liuxuzhao6@163.com
