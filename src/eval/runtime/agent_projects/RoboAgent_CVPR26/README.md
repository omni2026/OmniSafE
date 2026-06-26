# RoboAgent

This is the implementation of "[RoboAgent: Chaining Basic Capabilities for Embodied Task Planning](https://arxiv.org/abs/2604.07774)" (CVPR 2026).

Our model (a [Qwen2.5-VL-3B](https://github.com/qifan-han/Qwen2.5-VL) finetuned on [ALFRED](https://github.com/askforalfred/alfred)'s training set) is provided at [here](https://huggingface.co/woyut/RoboAgent_CVPR26).

## Evaluation on ALFWorld
Please prepare the environment as follows:
```sh
conda create -n RoboAgent_AW python=3.12
conda activate RoboAgent_AW
pip3 install torch==2.6.0 torchvision==0.21.0  --index-url https://download.pytorch.org/whl/cu124
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip3 install transformers==4.57.0 accelerate==1.7.0
pip3 install deepspeed==0.16.4
pip3 install peft==0.17.1
pip3 install qwen-vl-utils==0.0.14

pip3 install alfworld[full]
alfworld-download
```
Then you can run the evaluation on ALFWorld's visual environment with:
```sh
xvfb-run --server-num=99 -s "-screen 0 1280x1024x24" python run_aw.py --qwen_path /path/to/ckpt --save_path /path/to/save/images/and/logs
```

The code for evaluation on ALFWorld's textual environment will be updated soon.

## Evaluation on EmbodiedBench-ALFRED
Please build the environment (including the dataset) of EB-ALFRED following the official guidance of [EmbodiedBench](https://github.com/EmbodiedBench/EmbodiedBench). Then, we need to update a few packages in order to run the finetuned Qwen model.
```
pip3 install transformers==4.57.0 peft==0.17.1 jinja2==3.1.6
```
Then you can run the evaluation on EB-ALFRED with:
```sh
xvfb-run --server-num=99 -s "-screen 0 1280x1024x24" python run_ebalf.py --qwen_path /path/to/ckpt --save_path /path/to/save/images/and/logs --data_path /path/to/EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json --split base --server-num 99
```


## Citation
If you find this project useful, please kindly cite our paper:
```
@misc{RoboAgent,
      title={RoboAgent: Chaining Basic Capabilities for Embodied Task Planning}, 
      author={Peiran Xu and Jiaqi Zheng and Yadong Mu},
      year={2026},
      eprint={2604.07774},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2604.07774}, 
}
```