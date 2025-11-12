# vLLM Benchmark 开发落地计划

## 1. 总体目标
- 在单卡 RTX 4090 + uv 管理的 Python 环境下，完成 vLLM 基准测试框架。
- 支撑以下对比：纯 PyTorch baseline、vLLM Page Attention、Page Attention + 单项优化（FlashAttention 等）、全量优化组合。
- 负载基于 ShareGPT52K 与 Alpaca 分布混合，使用泊松过程模拟不同请求率；每档负载测试 1 分钟，整个 benchmark 流程 4 小时内可全部跑完。
- 核心指标：标准化时延（s/token）与并发批入请求数；支持多 trial 隔离与结果追溯。

## 2. 关键里程碑
1. **环境脚手架**：
   - 配置 uv 项目依赖（vLLM、transformers、flash-attn wheel、pydantic、typer 等）。
   - 构建统一的配置/日志目录结构（`results/<model>/<variant>/<timestamp>`）。
2. **数据与请求生成**：
   - 下载并缓存 ShareGPT52K 与 Alpaca 数据集，统计输入/输出长度分布。
   - 设计请求合成器：根据长度分布抽样 prompt/expected output token 目标。
   - 实现泊松过程请求调度器，可配置请求率、burstiness、测试时长。
3. **推理后端适配层**：
   - vLLM 客户端封装：OpenAI 兼容接口 + `VLLM_ATTENTION_BACKEND` 切换管理。
   - 纯 PyTorch baseline 服务：基于 transformers pipeline/Generate API，自实现 KV 缓存控制、批处理节流。
   - 统一请求回放接口，确保同一请求序列可在任意后端复现。
4. **实验 orchestrator**：
   - CLI/配置文件驱动，支持模型、variant、负载阶梯参数化。
   - 运行流程：启动后端 → 热身 → 正式压测 → 采集指标 → 清理。
   - 指标输出：raw 日志（每请求）、聚合 CSV/JSON、元信息（git hash、环境指纹）。
5. **分析与可视化**：
   - 生成标准化时延 vs 请求率、输出吞吐 vs 并发的图表脚本。
   - 汇总报告模板，便于快速对比不同优化方案。
6. **扩展：MoE 模型试验**（Qwen/Qwen3-30B-A3B-Instruct-2507-FP8）：在 dense 实验完毕后复用框架进行。

## 3. 详细任务拆解
- **环境与依赖**
  - [ ] 写 uv workflow 脚本或 Makefile，自动安装基础依赖与 flash-attn 轮子。
  - [ ] 封装 HuggingFace 镜像/缓存路径，避免重复下载。
- **数据处理**
  - [ ] 实现 `datasets/loader.py`：下载、缓存、统计分布、抽样。
  - [ ] 编写 `datasets/mix_profile.json`：长短提示混合比例、目标 token 分布。
- **负载模拟**
  - [ ] `loadgen/scheduler.py`：泊松到达、burstiness、自适应队列。
  - [ ] `loadgen/request_stream.py`：根据 profile 生成请求（prompt、max tokens、温度）。
  - [ ] 控制测试时长：1 分钟/档，支持自动截断。
- **后端封装**
  - [ ] `backends/vllm_runner.py`：启动/停止 vLLM 进程，配置 backend、page attention、flash-attn。
  - [ ] `backends/torch_runner.py`：纯 PyTorch 服务封装（异步推理、批处理、KV 缓存策略）。
  - [ ] 统一接口 `InferenceBackend`：`prepare() -> start() -> infer(requests) -> stop()`。
- **指标收集**
  - [ ] `metrics/collector.py`：记录请求 id、排队时间、TTFT、E2E latency、token 数。
  - [ ] 计算标准化时延、吞吐、批入请求数曲线。
  - [ ] 输出 JSON/CSV，并持久化到 trial 目录。
- **Orchestrator 与 CLI**
  - [ ] `runner/main.py`：解析配置、调度实验矩阵、并行/串行试验管理。
  - [ ] Trail 管理：基于时间戳 + UUID 创建目录，落盘配置与结果。
  - [ ] 错误恢复与重跑机制（可选，若时间允许）。
- **Dense 模型实验矩阵**
  - Variant 定义：
    1. `torch-baseline`
    2. `vllm-page`
    3. `vllm-page+flash-attn`
    4. `vllm-all-opt`（page + flash + 其他单项启用）
  - 请求率：单请求 + 多档并发（例如 1、5、10、20 req/s），每档 60s。
- **MoE 扩展**
  - 复用 orchestrator，调整 max concurrency、KV cache 配置。

## 4. 结果与文档
- 在 `results/README.md` 维护 trial 索引，包含：日期、模型、variant、配置文件路径。
- 汇总脚本将生成 `reports/<timestamp>.md`，附带关键曲线图（matplotlib 或 seaborn）。
- 主仓库 `README.md` 更新：运行前提、命令示例、flash-attn 安装指引、输出说明。

## 5. 风险与应对
- **FlashAttention 兼容性**：验证 wheel 与当前 torch/cuBLAS 版本；若冲突，回退到源码编译或禁用。
- **PyTorch baseline 性能过低**：提前设置合理的 max concurrency，避免 GPU OOM；必要时缩短请求率阶梯。
- **数据驱动耗时超限**：增加缓存与抽样机制，限制每档测试时间，确保 4 小时内完成。
- **日志数据量过大**：按 trial 归档并压缩原始日志，只保留聚合关键指标。

## 6. 时间安排（工作量预估）
1. 环境与基础设施：0.5 天
2. 数据与负载模块：0.5 天
3. 后端封装与 orchestrator：1.0 天
4. Dense 模型实验跑通：0.5 天
5. 文档整理与 MoE 验证（可选）：0.5 天

> 合计 ~3 天开发投入；单次完整 benchmark 在 4 小时内跑完。
