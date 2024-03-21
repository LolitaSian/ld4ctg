# Environment

A suitable environment can be created with the following commands. 
```bash
conda env create -f environment.yml
python -m spacy download en_core_web_sm
```

# Datasets



数据集都存放在 `datasets/` 目录下

# Training

训练使用脚本使用

`./scripts/diffusion/text_diffusion_{dataset}.sh`

其中可选`{roc, e2e, sst2, ag_news}`.

使用已存模型继续训练，脚本要增加参数

```bash
  --resume_training 
  --resume_dir <模型目录>
```

模型目录下要有`args.json`和`model.bin`两个文件。

## Evaluation

在测试集上测试模型效果使用

`scripts/diffusion/eval_text_diffusion.sh` 

 `--resume_dir` 传入模型目录


Different sampling configurations can be explored by changing the `{num_samples, sampling_timesteps, ddim_sampling_eta}` arguments. We utilize 1,000 random samples for computing the metrics in our work. Note that MAUVE scores computed with different numbers of samples are not directly comparable (see [here](https://github.com/krishnap25/mauve) for more information about MAUVE scores).

使用五个随机种子在测试集上测试模型的效果

`scripts/diffusion/test_eval_text_diffusion.sh` 

The only difference is that the `eval_test` flag is used instead of the `eval` flag. 

The `--resume_dir` argument will need to be updated as before.



---
双曲空间：
无效，后续实验。
/geoopt
/hyrnn
/poincare_embeddings
/poincare_glove
另外同名后缀p的文件