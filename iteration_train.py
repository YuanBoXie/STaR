import os
import sys
import json
import shutil
import argparse

def record_folder(cur_iter):
    """
    根据当前迭代次数生成文件夹路径。

    参数:
    - cur_iter (int): 当前迭代次数。

    返回:
    - str: 根据任务名、实验名和迭代次数生成的文件夹路径。
    """
    return f"{task}/{experiment_name}/{experiment_name}_{cur_iter}"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--no_prompt', action='store_true', help="Whether to remove prompts during eval")           # 不用 few-shot
    parser.add_argument("--base_epochs", type=float, default=1., help="Epochs for the first iteration")
    parser.add_argument("--add_epochs", type=float, default=0.2, help="Epochs to add each iteration")
    parser.add_argument("--few_shot_train", action='store_true', help="Whether to use few shot training")

    # 可以不设置 step add
    parser.add_argument("--steady_grow", action='store_true', help="Whether to use a fixed number of epochs")       # 稳定增长步数
    parser.add_argument("--start_steps", type=float, default=40., help="Steps for the first iteration")             # 第一次外循环的步数（不同外循环步数可能不同）
    parser.add_argument("--exponential_grow", action='store_true', help="Whether to use a fixed number of epochs")  # 不同的外循环，指数增加学习步数
    parser.add_argument("--add_steps", type=float, default=20., help="Steps to add each iteration")                 # add step 策略 steady_grow  配对参数
    parser.add_argument("--grow_steps", type=float, default=1.2, help="Steps to add each iteration")                # add step 策略 exponential_grow 配对参数
    
    # 使用合理化的 wrong exps 的比例
    parser.add_argument("--p_rationalization", type=float, default=1., help="Percent of wrong examples to rationalize")
    # [TODO]
    parser.add_argument("--p_show_hint_save", type=float, default=0., help="Percent of rationalization hints to save")	
    parser.add_argument('--rationalize', action='store_true', help="Whether to use rationalization")        # 使用合理化

    parser.add_argument("--start_iter", type=int, default=1, help="Starting iteration")
    parser.add_argument("--n_iters", type=int, default=64, help="Upper limit on outer loop iterations")     # 论文中的外循环迭代次数，重复多少次 STaR 微调方法
    parser.add_argument("--copy_n", type=int, default=0, help="Number of files to copy each iteration")
    parser.add_argument("--n_train_samples", type=int, default=10000, help="Number of training examples")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Batch size")            # [TODO]

    parser.add_argument("--task", type=str, default="commonsenseqa", help="Whether to run on arithmetic")   # 选择数据集/任务，论文中有 CommonsenseQA、GSM8K 分别是常识问答和数学计算
    parser.add_argument('--direct', action='store_true', help="Whether to use direct prediction, sans scratchpad")
    parser.add_argument("--gen_length", type=int, default=96, help="Length of generated output")
    parser.add_argument("--sequence_count", type=int, default=10, help="Sequences per batch on average")
    parser.add_argument("--base_model_location", type=str, default="gs://checkpoint-bucket/step_383500/", help="Finetuning ckpt")
    parser.add_argument('--dry_run', action='store_true', help="Whether to do a quick run to visualize output")
    parser.add_argument('--skip_eval', action='store_true', help="Whether to skip evaluation (e.g. arithmetic)")

    args = parser.parse_args()
    return args

def gen_train():
    train_cmd = f"python3 device_inference.py --config={prev_config} --split=train --gen_length={args.gen_length} --p_show_hint_save={args.p_show_hint_save} "
    if task != "commonsenseqa":
        train_cmd += f" --dataset_mode={task} "         # 因为里面默认是 commonsenseqa，并且代码里的值是 cqa，这里是兼容代码
    if args.rationalize:
        train_cmd += " --rationalize "
    if args.few_shot_train:
        train_cmd += " --few_shot_train "
    if cur_iter > 1 and args.no_prompt:
        train_cmd += f" --no_prompt --eval_seq {eval_seq} "
    train_cmd += f" --n_train_samples={args.n_train_samples} "
    train_cmd += f" >> result_logs/{experiment_name}.txt"
    print(f"Generating training set {cur_iter} using model {cur_iter - 1}: {train_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        if (cur_iter == 1) and os.path.exists(record_folder(0) + f"/{experiment_name}_0.txt"):
            print("First file cached")
        else:
            os.system(train_cmd)

def gen_records():
    gen_cmd = f'python3 create_finetune_tfrecords.py {record_folder(cur_iter - 1)} {record_folder(cur_iter - 1)}'
    print(f"Creating records for finetuning {cur_iter}: {gen_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        os.system(gen_cmd)
    train_set = f"{experiment_name}/{exp_iteration}.index"
    with open(f"data/{train_set}", "w") as new_data_file:
        new_data_file.write(f"{record_folder(cur_iter - 1)}.tfrecords")
    return train_set

def get_n_steps():
    """
    根据当前迭代次数和训练参数，计算当前迭代的步数。
    如果使用稳定增长策略，则步数为初始步数加上每次迭代增加的步数乘以当前迭代次数减一。
    如果使用指数增长策略，则步数为初始步数乘以增长因子幂次，幂次为当前迭代次数减一。
    如果不使用任何增长策略，则根据记录文件夹中的文件数量和每个文件中的数据点数量，计算当前迭代的步数。
    """
    if args.steady_grow:
        return int(args.start_steps + args.add_steps * (cur_iter - 1))              # 线性
    elif args.exponential_grow:
        return int(args.start_steps * (args.grow_steps ** (cur_iter - 1)))          # 指数
    else:
        # Count data points
        total_count = 0 # 上一次外循环数据点
        for cur_file in sorted(os.listdir(record_folder(cur_iter - 1)), key=lambda x: int(x.split('.')[0].split("_")[-1])):
            with open(f"{record_folder(cur_iter - 1)}/{cur_file}", encoding='utf-8') as train_file:
                train_file_text = train_file.read()
                total_count += len(train_file_text.split("\n\n"))
                print(len(train_file_text.split("\n\n")))
        train_epochs = args.base_epochs + args.add_epochs * (cur_iter - 1) # 当前外循环应有的 epochs 数（线性增长）
        cur_steps = int(total_count * train_epochs // (args.gradient_accumulation_steps * args.sequence_count))
        return cur_steps

def gen_config(train_set):
    """
    生成新的配置文件。
    读取基础配置文件，修改其中的模型目录、训练集、目标保存路径、总步数、名称、理性化概率和梯度累积步数。最后，它将修改后的配置写入新的配置文件，并返回配置文件的名称。
    """
    print(f"Creating new config file {cur_iter}")
    config_name = f'configs/{experiment_name}/{exp_iteration}.json'
    os.makedirs(record_folder(cur_iter), exist_ok=True)
    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)
        new_json["model_dir"] = f"strangeloop/{exp_iteration}"      
        new_json["train_set"] = train_set                                               # tfrecord
        new_json["target_save"] = record_folder(cur_iter) + f"/{exp_iteration}.txt"
        new_json["total_steps"] = get_n_steps()
        new_json["name"] = exp_iteration
        new_json["p_rationalization"] = args.p_rationalization
        new_json["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    with open(config_name, "w", encoding='utf-8') as new_json_file:
        json.dump(new_json, new_json_file, indent=2)
    return config_name

def train_model():
    model_cmd = f"python3 device_train.py --config {config_name} --tune-model-path={args.base_model_location}"
    print(f"Train model {cur_iter}: {model_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter):
        os.system(model_cmd)

def eval_model():
    eval_cmd = f"python3 device_inference.py --config={config_name} --split=dev --gen_length={args.gen_length} --p_show_hint_save={args.p_show_hint_save} "
    if task != "commonsenseqa":
        eval_cmd += f" --dataset_mode={task} "
    if args.no_prompt:
        eval_cmd += f" --no_prompt --eval_seq {eval_seq} "
    if args.few_shot_train:
        eval_cmd += " --few_shot_train "
    eval_cmd += f" >> result_logs/{experiment_name}.txt"
    print(f"Eval model {cur_iter}: {eval_cmd}")
    if not args.dry_run and (cur_iter >= args.start_iter) and not args.skip_eval:
        os.system(eval_cmd)

def copy_files():
    all_files = sorted(os.listdir(record_folder(cur_iter - 1)), key=lambda x: int(x.split('.')[0].split("_")[-1]))
    relevant_files = all_files[-args.copy_n:]
    for cur_file in relevant_files:
        shutil.copy(f"{record_folder(cur_iter - 1)}/{cur_file}", record_folder(cur_iter))

def make_first_config():
    """
    生成初始配置文件。
    这个函数读取基础配置文件，修改其中的目标保存路径和名称，然后，它将修改后的配置写回配置文件，并返回修改后的配置。
    """
    with open(prev_config, encoding='utf-8') as base_json_file:
        new_json = json.load(base_json_file)
        os.makedirs(record_folder(0), exist_ok=True)                                   # 生成外循环迭代的记录文件夹
        new_json["target_save"] = record_folder(0) + f"/{experiment_name}_0.txt"
        new_json["name"] = f"{experiment_name}_0"                                      # 当前迭代次数
        new_json["p_rationalization"] = args.p_rationalization
        new_json["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    with open(prev_config, "w", encoding='utf-8') as base_json_file:
        json.dump(new_json, base_json_file, indent=2)
    return new_json

if __name__ == "__main__":
    args = parse_args()
    print(args)
    task = args.task                                                                    # 选择数据集/任务：论文中有 CommonsenseQA、GSM8K
    experiment_name = "_".join(sys.argv[1:])                                            # 实验参数以_分割，拼接在一起命名
    experiment_name = ''.join(ch for ch in experiment_name if ch.isalnum() or ch == "_")# 确保 name 只有字母、数字、下划线（符合文件命名格式）
    if args.no_prompt:
        eval_seq = 128 + args.gen_length
    os.makedirs(f"configs/{experiment_name}", exist_ok=True)
    shutil.copy(f"configs/qa_base.json", f"configs/{experiment_name}/base.json")        # 复制一份实验配置模版
    prev_config = f"configs/{experiment_name}/base.json"                                # 实验配置模版的路径（后续代码会修改这个复制文件）
    new_json = make_first_config()

    os.makedirs(f'data/{experiment_name}', exist_ok=True)
    os.makedirs(f'{task}/{experiment_name}', exist_ok=True)
    os.makedirs(f'result_logs/', exist_ok=True)
    with open(f"result_logs/{experiment_name}.txt", "a+") as f:
        print("================================", file=f)                               # 类似 f.write
        print(args, file=f)
    for cur_iter in range(1, args.n_iters):                                             # 论文中的外循环迭代次数，重复多少次 STaR 微调方法
        exp_iteration = f"{experiment_name}_{cur_iter}"
        gen_train() # Generate the training set                                         # 第一次不执行
        train_set = gen_records() # Create the tfrecords from the data                  # "{experiment_name}/{exp_iteration}.index"
        config_name = gen_config(train_set) # Create the new configuration file         # 核心是修改 total_steps
        train_model() # Train the new model
        eval_model() # Evaluate the new model
        prev_config = config_name  # Prepare for next iteration
        if args.copy_n > 0:
            copy_files()                                                                # [TODO] 复制上次外循环的一些配置文件，暂时不知道有啥用
