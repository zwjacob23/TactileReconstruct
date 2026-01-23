import subprocess
import sys

python = sys.executable  # 保证用的是当前环境的 python
experiments = []

# -------- round 模式 --------
round_nmodes = [2, 3, 5]

for nmode in round_nmodes:
    experiments.append({
        "nmode": nmode,
        "smode": "round",
    })
# for nmode in round_nmodes:
#     experiments.append({
#         "nmode": nmode,
#         "smode": "subject",
#     })

# # -------- object 模式 --------
test_objects = [
    '水溶', '长方体', '海之言', '怡宝',
    '高脚杯', '三棱柱', '乌龙', '正方体',
    '薯片', '圆柱体', '红酒杯', '七喜'
]

# for nmd in round_nmodes:
#     for obj in test_objects:
#         experiments.append({
#             "nmode": nmd,
#             "smode": "object",
#             "testobj": obj,
#         })

for i, exp in enumerate(experiments):
    cmd = [python, "train.py"]
    for k, v in exp.items():
        cmd.append(f"--{k}")
        cmd.append(str(v))

    print(f"\n{'='*20} Running Experiment {i+1} {'='*20}")
    print("Command:", " ".join(cmd))

    subprocess.run(cmd, check=True)
