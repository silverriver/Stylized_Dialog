# Stylized Dialogue Response Generation Using Stylized Unpaired Texts

## Experiments regarding to the dataset TCFC

**TCFC** Dataset is a dataset aiming at generating formal responses according to informal posts. For more detail, please refer to the paper [A Dataset for Low-Resource Stylized Sequence-to-Sequence Generation](https://www.msra.cn/wp-content/uploads/2020/01/A-Dataset-for-Low-Resource-Stylized-Sequence-to-Sequence-Generation.pdf) and this [repo](https://github.com/MarkWuNLP/Data4StylizedS2S).

You may **train** and **evaluate** the model. To run the code, first clone all the files to a local folder (e.g. `text_style`). After that, create a `data` folder in `text_style`, and move all the data files to `text_style/data`. 

For TCFC Dataset, data files are as follows:

1. `tcfc_informal_dialogue_300k_url_filtered.txt`:
Paired dialogue data in the informal style.

2. `tcfc_formal_500k.txt`,
Unpaired texts in the formal style.

3. `tcfc_valid_1k.txt`, 
Testing dialogues in both formal and informal styles. 

## Training

1. Make a new project folder to store the datasets and models, and copy all the downloaded dataset files to a folder named `data`:

    ```bash
    mkdir stylized_dialog
    cd stylized_dialog   
    mkdir data  
    # copy all the downloaded dataset files to data
    ```

2. Download the pretrained [DialoGPT model](https://huggingface.co/microsoft/DialoGPT-small) to some local folder, e.g. `/home/handsomeboy/DialoGPT`

3. Copy the `config.json` file in `bt_beam` to the project folder `stylized_dialog`, 
and change the `gpt2_config_path`, `gpt2_tokenizer_vocab`, `gpt2_tokenizer_merges`, and `gpt2_weights_path`
fields in `config.json` according to your local folder (i.e., `/home/handsomeboy/DialoGPT`).

4. Use the following command to launch single GPU training:

    ```bash
    python3 train.py --config {path to your config} --gpu {your gpu}
    ```

    Use the following command to launch multi GPU training:

    ```bash
    bash run_training.sh  # remember to first modify run_training.sh to use correct config.json file and GPUs
    ```

    During training, prediction files will be output to the `stylized_dialog/eval` folder. The files can be used for evaluation. Model checkpoint files would be in `stylized_dialog/train`.

## Evaluation

1. Run `run_eval.sh` in `eval` directory. Specify the prediction file path as the `--eval_file_path` argument.

    ```bash
    bash run_eval.sh
    ```

    Note that you should train the `SVM` and `BERT` classifier first to calculate the `acc`. Please refer to the code in `eval/bert_eval_acc.py` and `eval/svm_eval_acc.py` for more details.
