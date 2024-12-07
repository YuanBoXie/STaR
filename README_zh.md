# STaR
- 本仓库是对 STaR: Bootstrapping Reasoning With Reasoning (NeurIPS 2022) 代码实现的一个中文解读版本。代码基于 [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) 构建，并包含 [this repo](https://github.com/albertqjiang/mesh-transformer-jax) 中的 masked training 代码。

- 入口点：`iteration_train.py`
- 关键代码：`device_train.py`, `device_inference.py`, and `create_finetune_tfrecords.py`

- 微调的实验模型：GPT-J-6B，一个 60 亿参数的自回归文本生成模型，在 [The Pile](https://pile.eleuther.ai/) 上训练。
- 实验环境：为 TPU 设计的代码，运行在 [TPU Research Cloud](https://sites.research.google/trc/) 上。
    - https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms
    - [TPU-VM architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)

软件依赖：
- 代码需要特定 JAX 版本 来运行 v1 models (eg. GPT-J 6B)： `jax==0.2.12`, `jaxlib==0.1.68` 否则会出现 cryptic xmap errors
- v2 模型可以用新的 jax 版本

微调：
- 在 TPU VM (a TPU v3-8) 上运行 `device_train.py`，可以以 ~5000 tokens/second 的速率微调，对于小到中型数据集够用了。 

# Architecture and Usage

大多数脚本旨在启动 TPU、通过 SSH 连接到其中以设置依赖项并从本地目录复制代码，然后启动可以接受 RPC 调用的 [Ray](https://github.com/ray-project/ray.git) 工作线程。

TPUVM 处理运行模型训练步骤和评估、检查点保存和加载，而驱动程序 python 程序处理数据加载和一般编排（例如何时保存检查点等）。这意味着大多数脚本（ train.py 、 eval_harness.py等）预期在与 TPU 相同区域的 GCE 虚拟机上运行，​​以最大限度地减少 RPC 延迟和数据传输成本。

其他脚本（通常是不带--tpu参数的脚本，例如device_sample.py 、 device_serve.py或device_train.py ）期望直接在 TPUVM 上运行。 

device_* 脚本仅适用于 v3-8 ，不适用于较大的 pod。

此外，还有一个示例 (resharding_example.py )，说明如何将提供的检查点（在 GPT-J-6B 的情况下有 8 个分片）转换为较小的数量，例如在 GPU 上运行时。

# Mesh Transformer JAX

haiku 库使用 JAX 中的 `xmap`/`pjit` 算子来进行 transformers 模型并行。并行机制类似：
[original Megatron-LM](https://arxiv.org/abs/1909.08053), 由于 speed 2d mesh 网络，它在 TPU 上非常高效。还有一个实现 [ZeRo style
sharding](https://arxiv.org/abs/1910.02054) 的实验模型版本。

该库设计用于在 TPUv3 上扩展至约 40B 参数，超出此范围应该使用不同的并行策略。如：[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) or [DeepSpeed](https://github.com/microsoft/DeepSpeed)。未来的研究方向之一是将这个代码库与 [swarm-jax](https://github.com/kingoflolz/swarm-jax) ，通过管道并行(pipeline parallelism)实现进一步的可扩展性。

# Model Details

| Hyperparameter    | Value  |
|-------------------|--------|
| n_parameters      | 6,053,381,344 |
| n_layers          | 28*    |
| d_model           | 4,096  |
| d_ff              | 16,384 |
| n_heads           | 16     |
| d_head            | 256    |
| n_ctx             | 2,048  |
| n_vocab           | 50,257 (same tokenizer as GPT-2/3)  |
| position encoding | [Rotary position encodings (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE dimensions   | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |

`*` 每层由一个前馈块和一个自注意力块组成

该模型由 28 层组成，模型维度为 4096，前馈维度为 16384。模型维度分为 16 个头，每个头的维度为 256。每个头的 64 个维度应用了旋转位置编码（RoPE） 。该模型使用与 GPT-2/GPT-3 相同的 BPE 集，使用 50257 的 tokenization vocabulary 进行训练。