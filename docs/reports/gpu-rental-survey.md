# GPU 租用调研：自主实时视频超分（VSR）项目

- 价格核查日期：**2026-10-14（Asia/Shanghai）**，约 23:30–23:45 CST
- 本地基线：Windows PC `dscs2pc` 上的 RTX 3070 8GB（用户在场时可用）
- 规则：只列我实际抓取到的数字；没有核实的项目标注为 **未核实**。价格随时会变，租用前请重新查看实时页面。

## 1. 本次核查到的价格

| 平台 | GPU | 价格 | 计费方式 / 备注 | 来源（2026-10-14 核查） |
|---|---|---|---|---|
| RunPod | RTX 3090 24GB | Community $0.22/hr · Secure $0.50/hr | Pod，可按秒计费 | runpod.io/pricing（页面标注 2026-10-08 更新） |
| RunPod | RTX 4090 24GB | Community $0.34/hr · Secure $0.74/hr（卡片显示 $0.89） | 同上 | 同上 |
| RunPod | RTX A6000 48GB | Community $0.33 · Secure $0.53/hr | | 同上 |
| RunPod | RTX A5000 24GB | Community $0.16 · Secure $0.27/hr | 便宜的推理/冒烟测试卡 | 同上 |
| RunPod | L4 24GB | $0.44 / $0.49/hr | 推理基准测试 | 同上 |
| RunPod | A100 80GB PCIe / SXM | $1.19 / $1.39（Community），$1.59（Secure） | | 同上 |
| RunPod | H100 PCIe / SXM 80GB | $1.99 / $2.69（Community），$2.89 / $3.49（Secure） | | 同上 |
| RunPod | 存储 | 容器盘 $0.10/GB/月；数据卷运行时 $0.10、闲置 $0.20/GB/月；网络存储 $0.07/GB/月（<1TB） | | 同上 |
| Vast.ai | RTX 4090（单卡） | 49 个可租报价：最低 **$0.136/hr**，中位数 **$0.561/hr** | 市场定价，按秒计费，有可中断（interruptible）档；页面称可中断"便宜 50%+" | console.vast.ai 公开 bundles API + vast.ai/pricing |
| Vast.ai | 3090 / A100 / H100 / A6000 | **未核实**（API 被限流） | | — |
| Lambda | A6000 48GB | $1.09/GPU/hr | 另加税 | lambda.ai/pricing |
| Lambda | A100 40GB（PCIe/SXM） | $1.99/hr；A100 80GB SXM（8 卡）$2.79/GPU/hr | | 同上 |
| Lambda | H100 SXM 单卡 | $4.29/hr；H100 PCIe $3.29/hr | | 同上 |
| AWS | H100（p5.4xlarge，1 卡，含东京） | $5.970/hr（Capacity Block 预留费） | 需预约；下次调价 2027 年 1 月 | aws.amazon.com/ec2/capacityblocks/pricing |
| AWS | A100（p4d 8 卡） | $1.696/GPU/hr（只能整台 8 卡，$13.57/hr） | Capacity Block，仅美国区 | 同上 |
| AutoDL | RTX 4090 24GB | **¥1.88/hr**（会员 95 折） | 按秒计费，最低 ¥0.01；人民币支付 | autodl.com 首页（搜索快照，2026-10-14 获取；直接抓取页面时只返回 JS 外壳） |
| AutoDL | RTX 3090 24GB | **¥1.32/hr**（会员 95 折） | 同上 | 同上 |
| AutoDL | A100 / H 系列 | **未核实**（第三方文章写 A100 80GB 约 ¥5.98/hr；未经确认，不作为价格锚点） | | — |
| 恒源云、矩池云 | — | **未核实**（页面是 JS 渲染，抓不到价格） | 第三方 CSDN 文章称恒源云 4090 约 ¥1.32/hr，未经确认 | — |
| 阿里云 PAI / 腾讯云 / 火山引擎 / GCP / Azure / Paperspace / CoreWeave | — | **未核实**（本次没有抓取） | 预计比上面的平台更贵；需要时再查 | — |

汇率换算（仅为估算）：用 ¥7.1/$ 粗略换算，AutoDL 4090 ¥1.88 ≈ $0.26/hr，与 RunPod Community 和 Vast 中位价处于同一档。

## 2. 定性对比（适用于本项目）

| 维度 | AutoDL（及其他国内平台） | Vast.ai | RunPod | Lambda / AWS |
|---|---|---|---|---|
| 从深圳访问的延迟 | 最好（国内节点） | 取决于主机，可筛选亚洲主机 | 美国/欧盟为主，SSH 有延迟 | 美国区；AWS 有东京 |
| 支付方式 | 支付宝/微信，人民币 | 信用卡/加密货币 | 信用卡/加密货币 | 信用卡，企业账户 |
| 最低计费 | 按秒 | 按秒 | 按秒/按小时 | 按小时；AWS 为预约区块 |
| 访问方式 | SSH + JupyterLab，可选 PyTorch 镜像 | SSH/Jupyter，Docker 镜像 | SSH/Jupyter，模板 | SSH，Lambda Stack |
| PyTorch / CUDA | 预置镜像 | 自带 Docker（如 `nvcr.io/nvidia/pytorch`） | 官方 PyTorch 模板 | 预装 |
| TensorRT | 可 pip 安装 `tensorrt`，或使用 NGC 镜像（需验证网络/镜像访问） | 用 NGC TensorRT 容器，最简单 | 用 NGC 容器 | 用 NGC 容器 |
| 数据传输 / 出站流量 | 国内传输快；有网盘 | 按主机计费，各异 | 未见明确出站流量费用（未核实） | AWS 出站流量收费 |
| 隐私 | 数据留在国内（PIPL）；共享宿主机 | 社区主机：**不要放敏感数据** | Secure Cloud 比 Community 更好 | 最强 |
| 中断风险 | 卡可能被占满、关机后可能租不回同一张卡 | 可中断档会被抢占 | Community 可能断开 | 低 |

对 VSR 数据的影响：公开数据集（REDS、Vimeo-90K）用哪家都可以；用户自己的私有视频只放在 AutoDL、RunPod Secure 或 Lambda 上。

## 3. 推荐的入门方案

1. **用户不在时跑短训练 / 消融实验** → **AutoDL RTX 4090（¥1.88/hr）**，国外备选 RunPod Community 4090（$0.34/hr）。24GB 显存允许用更大的 patch 和 batch；按秒计费；每 N 步保存一次检查点到持久化磁盘。备选：AutoDL 3090（¥1.32/hr）。
2. **大型教师模型训练 / 蒸馏** → **RunPod A100 80GB（Community $1.19–1.39/hr，Secure $1.59）**，或者 AutoDL 上同等的 A100（价格需先核实）。只有当某次运行确实受计算瓶颈限制且时间紧时，才考虑 H100（RunPod $1.99–2.69）。在有迹象表明需要多卡之前，不要上多卡。
3. **TensorRT FP16 测速** → 主要的延迟目标应该就是用户本地的 RTX 3070（真正的部署目标卡），**不需要租**。如果需要跨卡对比，可以租 RunPod L4（$0.44）或 A5000（$0.16）一小时，使用 NGC TensorRT 容器。

### 月度费用估算（根据上表抓取价格推算，**仅为估算**）
- **轻度使用**：40 小时 4090 + 10 小时 A100 → AutoDL 约 ¥75 + A100 约 $12–16 ≈ **¥160–200/月（约 $25–30）**
- **重度使用**：200 小时 4090 + 80 小时 A100 80GB + 存储 → 4090 约 ¥376（RunPod Community 约 $68），A100 约 $95–127，存储（200GB × $0.07）约 $14 ≈ **¥1,300–1,800/月（约 $180–250）**
- 只在 RunPod Secure 上跑同样的重度负载：200×$0.74 + 80×$1.59 ≈ $275 + 存储

## 4. 下一步建议
暂时不租。当某个实验在 3070 上的预计运行时间超过约 12 小时，或者需要超过 8GB 显存（大 patch、教师模型），且用户不在时，再注册 AutoDL（人民币支付、延迟最低），先充值少量金额（例如 ¥50），在 4090 上跑 1 小时冒烟测试：验证训练脚本 + 检查点保存/恢复，然后再排长任务。

## 5. 注意事项
- Vast 只核实了 4090 的价格；其他型号的 API 请求被限流。
- AutoDL 价格来自官方首页的搜索快照（直接抓取只返回 JS 页面）；租用前请在控制台确认。
- 所有标注"未核实"的内容，在依赖之前都要先检查。
