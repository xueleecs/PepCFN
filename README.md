# PepCFN

1.embeding
1.1 For Bert
firstly, you need to download pytorch_model.bin file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd/blob/main/pytorch_model.bin. And put pytorch_model.bin file into prot_bert_bfd directory.

The main program in the train folder protBert_main.py file. You could change the load_config function to achieve custom training and testing, such as modifying datasets, setting hyperparameters and so on. File protBert_main.py has detail notes.

The project is mainly implemented through Pytorch and sklearn. See requirements.txt for details of dependent packages.

1.2 For ESM
refer to https://huggingface.co/docs/transformers/model_doc/esm
https://github.com/facebookresearch/esm

2. run
python main.py

If you have other questions, please send mail to Xueleecs@gmail.com.
