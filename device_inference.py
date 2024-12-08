import argparse
import json
import time

import jax
import numpy as np
import optax
import random
import wandb

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer
from smart_open import open as smart_open
import transformers
import tensorflow_datasets as tfds
from mesh_transformer.util import clip_by_global_norm
import jsonlines
import pprint
from tqdm import tqdm

basic_open = open
pp = pprint.PrettyPrinter(indent=2).pprint


def eval_output(output, answers, context, example_classes, accuracy, target_save, tokenizer, show=False, direct=False, endoftext="<|endoftext|>"):
    """
    评估输出结果，统计准确率，并将成功的示例保存到指定文件中。

    参数:
    - output (list): 模型的输出结果。
    - answers (list): 正确答案列表。
    - context (list): 上下文列表。
    - example_classes (list): 示例类别列表。
    - accuracy (dict): 用于统计准确率的字典。
    - target_save (str): 成功示例保存的文件路径。
    - tokenizer (transformers.PreTrainedTokenizer): 用于处理文本的分词器。
    - show (bool, optional): 是否打印成功示例到控制台。默认为 False。
    - direct (bool, optional): 是否使用直接预测，跳过scratchpad。默认为 False。
    - endoftext (str, optional): 用于标记文本结束的字符串。默认为 "<|endoftext|>"。

    返回:
    - list: 成功示例的索引列表。
    """
    successful_examples = []
    enum_outputs = enumerate(output[1][0][:, :, 0])
    for (idx, o), target, cur_base_context, example_class in zip(enum_outputs, answers, context, example_classes):
        cur_output = tokenizer.decode(o)
        output_numbers = cur_output.split('\n')
        if example_class not in accuracy:
            accuracy[example_class] = {'accurate': 0, 'total': 0}
        accuracy[example_class]['total'] += 1
        if len(output_numbers) == 0:
            continue
        try:
            if args.dataset_mode == "cqa":
                output_numbers = output_numbers[0]
                if "<|endoftext|>" in output_numbers:
                    output_numbers = output_numbers.split("<|endoftext|>")[0]
                output_prediction = output_numbers[-3]                                  # 选项
            elif args.dataset_mode == "gsm":
                output_prediction = ""
                for line_idx, line in enumerate(output_numbers):
                    if "####" in line:
                        output_numbers = "\n".join(output_numbers[:line_idx + 1])
                        if "<|endoftext|>" in output_numbers:
                            output_numbers = output_numbers.split("<|endoftext|>")[0]
                        output_prediction = output_numbers.split("####")[-1].strip()
                        break
            elif args.dataset_mode == "arithmetic":
                if len(output_numbers) == 0:
                    continue
                elif "<|endoftext|>" in output_numbers:
                    prediction_index = output_numbers.index("<|endoftext|>") - 1
                elif "</scratch>" in output_numbers:
                    prediction_index = output_numbers.index("</scratch>") + 1
                    if prediction_index == len(output_numbers):
                        continue
                else:
                    if direct and len(output_numbers) > 1:
                        prediction_index = 1
                    else:
                        prediction_index = 0
                output_prediction = output_numbers[prediction_index]                      # 计算结果

            if "<|endoftext|>" in output_prediction:
                output_prediction = output_prediction.split("<|endoftext|>")[0]

            correct = output_prediction.lower() == target.lower()                         # 判断输出是否和目标一致
            if correct:
                accuracy[example_class]['accurate'] += 1                                  # 回答正确，计数++
                with basic_open(target_save, 'a+') as new_train_f:
                    if args.dataset_mode == "cqa" or args.dataset_mode == "gsm":
                        new_example = cur_base_context + output_numbers + endoftext       # 正确回答的样本作为新的训练样本
                    elif args.dataset_mode == "arithmetic":
                        if args.few_shot_train:
                            raise NotImplementedError
                        joined_output = "\n".join(output_numbers[:prediction_index + 1])
                        if "<|endoftext|>" in joined_output:
                            joined_output = joined_output.split("<|endoftext|>")[0]
                        new_example = cur_base_context + joined_output + endoftext       # 正确回答的样本作为新的训练样本
                    if show:
                        print(new_example)
                    print(new_example, file=new_train_f, end="")                         # 把回答正确的样本写入文件中
                successful_examples.append(idx)
        except IndexError:
            pass
    return successful_examples

def get_score(subcounts):
    if subcounts['total'] == 0:
        return 0
    return subcounts['accurate'] / subcounts['total']

def question_to_context(data_example, hint=False, dataset_mode='cqa', direct=False):
    """"
    将问题转为 prompt

    - hint: 是否开启合理化
    """
    if dataset_mode == 'cqa':
        context = f"Q: {data_example['question']['stem']}\nAnswer Choices:\n"
        for choice in data_example['question']['choices']:
            if hint and (choice['label'].lower() == data_example['answerKey'].lower()):
                context += f"({choice['label'].lower()}) {choice['text']} (CORRECT)\n"
            else:
                context += f"({choice['label'].lower()}) {choice['text']}\n"
        context += "A:"
    elif dataset_mode == 'gsm':
        context = f"Q: {data_example['question']}"
        if hint:
            chosen_hint = data_example['answer']                # gsm 竟然直接把答案作为 hint
            context += f" ({chosen_hint})"
        context += "\nA:"
    elif dataset_mode == "arithmetic":
        context = ""
        for example_split, next_example_split in zip(data_example.split('Target:')[:-1], data_example.split('Target:')[1:]):
            if direct and "</scratch>" in example_split:
                context += example_split.split("</scratch>")[-1]
            else:
                context += example_split
            context += "Target:"
            if hint:
                context += " " + next_example_split.split("\n")[-5]
    return context


def examples_to_batch(data_examples, few_shot_prompts, seq, tokenizer, hint=False, direct=False, p_show_hint_save=0.1):
    batch = {
        "base_context": [],
        "initial_batch": [],
        "lengths": [],
        "padded_batch": [],
        "answers": [],
        "classes": []                                   # 分类
    }
    for data_class, data_example in data_examples:
        batch['classes'].append(data_class)
        # Context, without the few-shot prompt
        hintless_base_context = question_to_context(data_example, hint=False, dataset_mode=args.dataset_mode, direct=direct)    # 不带 hint
        base_context = question_to_context(data_example, hint=hint, dataset_mode=args.dataset_mode, direct=direct)
        if args.dataset_mode == "arithmetic":
            few_shot_prompts = base_context.split("\n\n")[:-1]
            base_context = base_context.split("\n\n")[-1]
            hintless_base_context = hintless_base_context.split("\n\n")[-1]

        if random.random() < p_show_hint_save:  # 默认是 0
            hintless_base_context = base_context

        # We always want to act as if no hint was given
        if args.few_shot_train:
            if args.dataset_mode == "arithmetic":
                raise NotImplementedError
            else:
                save_context = "\n\n".join(commonsense_prompts) + "\n\n"
                save_context += hintless_base_context
                batch['base_context'].append(save_context)
        else:
            batch['base_context'].append(hintless_base_context)

        # Input tokens
        if args.no_prompt:
            context = ""
        else:
            context = "\n\n".join(few_shot_prompts) + "\n\n"            # 最终prompt部分 1：默认带 few-shot

        context += base_context                                         # 最终prompt部分 2：当前问题（可能带有合理化）
        tokens = tokenizer.encode(context)                              # tokenizer
        batch['initial_batch'].append(tokens)
        # Input lengths
        batch['lengths'].append(len(tokens))
        # Padded tokens
        provided_ctx = len(tokens)
        pad_amount = max(seq - provided_ctx, 0)                         # seq 是最大窗口长度，如果不够这个长度需要 pad
        if provided_ctx > seq:
            tokens = tokens[-seq:]                                      # 如果超出，需要截断
        batch['padded_batch'].append(np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32))

        # Answer
        if args.dataset_mode == "arithmetic":
            if len(data_example.split("\n")) >= 3:
                target = data_example.split("\n")[-3]
            else:
                target = "invalid"
        elif args.dataset_mode == "cqa":
            target = data_example['answerKey']
        elif args.dataset_mode == "gsm":
            target = data_example['answer'].split("#### ")[-1]
        batch['answers'].append(target)
    batch["lengths"] = np.asarray(batch["lengths"], dtype=np.uint32)
    batch["padded_batch"] = np.array(batch["padded_batch"])
    return batch


def eval_batch(examples, few_shot_prompts, seq, tok, gen_length, gen_params, accuracy, target_save, hint=False, direct=False):
    batch = examples_to_batch(examples, few_shot_prompts, seq, tok, hint=hint, direct=direct, p_show_hint_save=args.p_show_hint_save)   # 把example批处理成合适的prompt
    output = network.generate(batch["padded_batch"], batch["lengths"], gen_length, gen_params)    # 实际上执行输出的代码
    return eval_output(                                                                           # 评估输出结果，记录回答正确的样本
        output, batch["answers"], batch["base_context"], batch["classes"], accuracy, target_save, tok, direct=direct
    )


def load_model(params, ckpt_path, devices, mesh_shape):
    network = CausalTransformer(params)
    start = time.time()
    network.state = read_ckpt(network.state, ckpt_path, devices.shape[1])
    print(f"{ckpt_path} network loaded in {time.time() - start:.06}s on {jax.device_count()} devices")
    local_shards = max(jax.local_device_count() // mesh_shape[1], 1)
    del network.state["opt_state"]
    network.state = network.move_xmap(network.state, np.zeros(local_shards))
    return network

def eval_examples(data_examples, few_shot_prompts, few_shot_prompts_hint, direct=False):
    accurate_count = {}
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    main_examples, hint_examples = [], []
    pbar = tqdm(data_examples, smoothing=0)
    for data_example in pbar:   # 逐个遍历：而单个样本的执行和合理化样本的执行都是 cache 到一个 batch 再执行
        main_examples.append(data_example)
        if len(main_examples) == args.eval_batch_size:  # 默认值 8
            successful_examples = eval_batch(           # 评估
                main_examples, few_shot_prompts, seq, tokenizer,
                args.gen_length, gen_params, accurate_count, target_save, direct=direct
            )
            for example_idx, example in enumerate(main_examples):
                if (example_idx not in successful_examples) and (random.random() < params.get('p_rationalization', 1.)): # p_rationalization 默认值是 1
                    hint_examples.append(example)   # 如果回答失败，加入 hint 合理化样本中
            main_examples = [] # 清空队列

        if args.rationalize and len(hint_examples) >= args.eval_batch_size: # 合理化
            cur_hint_examples = hint_examples[:args.eval_batch_size]
            cur_hint_examples = [                                           # hint 样本修改 key
                (hint_example_key + "_r", hint_example) for hint_example_key, hint_example in cur_hint_examples
            ]
            eval_batch(                                                     # 评估
                cur_hint_examples, few_shot_prompts_hint, hint_seq, tokenizer,
                args.gen_length, gen_params, accurate_count, target_save, hint=True, direct=direct  # 开启 hint 合理化
            )
            hint_examples = hint_examples[args.eval_batch_size:]            # 清空当前合理化的样本
        pbar.set_description(f"{split} " + ", ".join([
            f"{cur_key}: {get_score(cur_counts):0.4f}" for cur_key, cur_counts in accurate_count.items()
        ]))
    return accurate_count

def get_ckpt_path(params, ckpt_step=-1):
    bucket = params["bucket"]
    model_dir = params["model_dir"]
    if ckpt_step == -1:
        ckpt_step = params["total_steps"]
    return f"gs://{bucket}/" + (f"step_{ckpt_step}/" if ckpt_step > 10000 else f"{model_dir}/step_{ckpt_step}/")

def set_opt(params):
    """
    根据给定的参数设置优化器。
    """
    params["sampler"] = nucleaus_sample                                 # 采样方法，来自 mesh transfomer
    opt = optax.chain(
        optax.scale(1 / params.get("gradient_accumulation_steps", 1)),  # 梯度累积的步数，因此要除这个因子
        clip_by_global_norm(1),                                         # 最大不能超过1
        optax.scale_by_adam(),                                          # 添加 Adam 优化器的缩放操作
        optax.additive_weight_decay(0),                                 # 加性权重衰减，在当前配置中没有应用权重衰减
        optax.scale(-1),                                                # 将梯度乘以 -1，用于实现梯度下降
        optax.scale_by_schedule(util.gpt3_schedule(0, 1, 0, 0))         # 添加一个基于调度的缩放操作
    )
    params["optimizer"] = opt

def get_dataset(args):
    if args.dataset_mode == "cqa":
        with jsonlines.open(f'commonsenseqa/{split}_rand_split.jsonl') as reader:
            dataset = [("cqa", example) for example in reader]
    elif args.dataset_mode == "gsm":
        with jsonlines.open(f'gsm/{split}_rand_split.jsonl') as reader:
            dataset = [("gsm", example) for example in reader]
    elif args.dataset_mode == "arithmetic":
        digit_range = list(range(1, 6))
        dataset = []
        for i in digit_range:
            with basic_open(f'arithmetic/train_scratch/{i}.txt') as f:
                dataset += [(str(i), example) for example in f.read().split('<|endoftext|>')]

    if split == "train":
        random.shuffle(dataset)
        dataset = dataset[:args.n_train_samples]
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Config file location")
    parser.add_argument('--direct', action='store_true', help="Whether to use direct prediction, sans scratchpad")
    parser.add_argument('--rationalize', action='store_true', help="Whether to use rationalization")
    parser.add_argument('--no_prompt', action='store_true', help="Whether to remove prompts during eval")
    parser.add_argument('--few_shot_train', action='store_true', help="Whether to remove few-shot-prompts during train")
    parser.add_argument('--show_hint_prompt', action='store_true', help="Whether a hint prompt will be necessary")
    parser.add_argument("--split", type=str, default="dev", help="Split")
    parser.add_argument("--dataset_mode", type=str, default="cqa", help="Which dataset to run on")
    parser.add_argument("--n_train_samples", type=int, default=3000, help="Number of training examples")
    parser.add_argument("--gen_length", type=int, default=96, help="Generation length")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Size of batches in eval")
    parser.add_argument("--p_show_hint_save", type=float, default=0.0, help="Percent of rationalization hints to save")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Which checkpoint to eval. -1 means the final one")
    parser.add_argument("--eval_seq", type=int, default=-1, help="Sequence length. -1 means the one in the param file")

    args = parser.parse_args()
    return args

def transform_example(example):
    new_example = {
        "question": example["english_text"],
        "answer": "#### " + example["ans"]
    }
    return new_example

if __name__ == "__main__":
    # 参数解析
    args = parse_args()
    print(args)
    split = args.split                              # 'dev'
    params = json.load(smart_open(args.config))     # smart_open 是一个用于打开文件的函数，支持多种文件格式和存储后端，本地文件，aws s3，gcs 等等

    # 初始化 wandb
    project = params.get("wandb_project", "mesh-transformer-jax")               # 日志服务所属的项目，随便什么值，这里不重要
    experiment_details = params["name"].split("_")
    wandb_name = "_".join(experiment_details[:-1])
    wandb_iteration = int(experiment_details[-1])
    wandb.init(project=project, name=wandb_name, config=params, resume=True)    # resume=True: 表示如果有相同名称的实验已经存在，则恢复该实验的状态，而不是创建一个新的实验。

    # 根据配置加载不同的 prompt 设置
    prompts_file = "prompts.txt" if not args.direct else "prompts_direct.txt"   # 默认不带 direct，即用带 few-shot 和 rationales 的 prompt
    prompts_file = f"{args.dataset_mode}/{prompts_file}"                        
    if args.no_prompt:
        commonsense_prompts = []
    else:
        with basic_open(prompts_file) as prompts:
            commonsense_prompts = prompts.read().split("\n\n")
    prompts_hint_file = "prompts_answer_key.txt" if not args.direct else "prompts_direct_answer_key.txt"
    prompts_hint_file = f"{args.dataset_mode}/{prompts_hint_file}"
    if args.no_prompt and not args.show_hint_prompt:
        commonsense_prompts_hint = []
    else:
        with basic_open(prompts_hint_file) as prompts:
            commonsense_prompts_hint = prompts.read().split("\n\n")

    # 参数设置
    per_replica_batch = params["per_replica_batch"]                             # 数据并行参数：1
    cores_per_replica = params["cores_per_replica"]                             # 模型并行参数：模型并行中的每个 replica 的核心数，默认是 8
    target_save = params["target_save"] if split != "dev" else f'{args.dataset_mode}/new_dev.txt'
    seq = params["seq"] if args.eval_seq == -1 else args.eval_seq
    hint_seq = seq
    set_opt(params)

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)   # (replica 数量，每个 replica 的核心数)
    devices = np.array(jax.devices()).reshape(mesh_shape)                       # 为每个 replica 划分 cores，形成一个资源分配矩阵
    ckpt_path = get_ckpt_path(params, args.ckpt_step)                           # 默认用最新的 ckpt
    with jax.experimental.maps.mesh(devices, ('dp', 'mp')):                     # 并行策略的维度：dp，数据并行，mp，模型并行
        network = load_model(params, ckpt_path, devices, mesh_shape)

        dataset = get_dataset(args)
        dataset_keys = set([datakey for datakey, _ in dataset])

        total_batch = per_replica_batch * jax.device_count() // cores_per_replica * args.eval_batch_size    # 数据并行侧，一次性输入的数据 batch 大小
        gen_params = {"top_p": np.ones(total_batch) * 0.9, "temp": np.ones(total_batch) * 0.01}             # top_p: 控制生成文本的多样性的一种采样策略, Nucleus Sampling; temp: 温度参数，用于控制生成文本的随机性。温度越高，生成的文本越随机；温度越低，生成的文本越确定。

        accurate_count = eval_examples(dataset, commonsense_prompts, commonsense_prompts_hint, direct=args.direct)
        for cur_key, cur_counts in accurate_count.items():
            print(f"{split}, {cur_key}, {get_score(cur_counts)}")
            wandb.log({f"{split}_{cur_key}_accuracy": get_score(cur_counts), "iteration": wandb_iteration})
