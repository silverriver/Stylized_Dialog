# Stylized Dialogue Response Generation Using Stylized Unpaired Texts

## Experiments regarding to the dataset WDJN

**WDJN** stands for Weibo Dialogues and Jinyong's Novels. It contains 300K single-turn dialogues collected from Weibo and 95.13K utterances extracted from Jinyong's novel. Please refer to [our paper](https://arxiv.org/abs/2009.12719) for more details about this dataset. 

You may **train**, **infer** with and **evaluate** the model. The data files in WDJN are listed as follow. All files are in the `tsv` format with the first column representing the style, 0 for style S0 (Weibo style), and 1 for style S1 (Jinyong style):

1. `crowded_300k.txt`: 
Paired dialogue data collected from Weibo.

2. `jinyong_dialogue_text_style.txt`:
Unpaired texts extracted from Jinyong's novel.

3. `valid_2k.txt`:
Testing dialogues extracted from Weibo

4. `jinyong_2k_eval.txt`:
Testing dialogues extracted from Jinyong's novel. The last two columns of this file demonstrate the original sentences from which these dialogues are extracted.

## Training

1. Make a new project folder to store the datasets and models, and copy all the downloaded dataset files to a folder named `data`:

    ```bash
    mkdir stylized_dialog
    cd stylized_dialog   
    mkdir data  
    # copy all the downloaded dataset files to data
    ```

2. Download the pretrained GPT model to some local folder, e.g. `/home/handsomeboy/chinese_gpt_original`

3. Copy the `config.json` file in `bt_beam` to the project folder `stylized_dialog`, 
and change the `vocab_path` and `cgpt_parameters_dir` fields in `config.json` according to your local folder (i.e., `/home/handsomeboy/chinese_gpt_original`).

4. Use the following command to launch single GPU training:

    ```bash
    python3 train.py --config {path to your config} --gpu {your gpu}
    ```

    Use the following command to launch multi GPU training:

    ```bash
    bash run_training.sh  # remember to first modify run_training.sh to use the correct config.json file and GPUs
    ```

Note that the eval steps in the training process will take a long time to infer the whole valid dataset to calcuate BLEU. You may want to use a small valid dataset in the training process.

Unlike TCFC Dataset implementation, prediction files will not be generated. You should go through the **inference** phase to obtain prediction files beforing runing the evaulation scripts.

## Inference

Run `run_infer.sh` scirpt to infer from the save model. The `--gpu` argument specifies which GPU to be used (should only specify one GPU). The `--ckpt` argument specifies which checkpoint file in the folder `stylized_dialog/train` to be loaded.

    ```bash
    bash run_infer.sh
    ```

After inference, prediction files will be output to the `eval` folder under your project folder (e.g. `stylized_dialog/eval`).

## Evaluation

1. Run `run_eval.sh` in the `eval` directory to evaluate the model. Specify the prediction file path as the `--eval_file_path` argument.

    ```bash
    bash run_eval.sh
    ```

    Note that you should train the `SVM` and `BERT` classifier first to calculate the `acc`. Please refer to the code in `eval/bert_eval_acc.py` and `eval/svm_eval_acc.py` for more details.
