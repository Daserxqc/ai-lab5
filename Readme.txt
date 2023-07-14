执行代码所需的环境用了pip freeze > requirements.txt，见requirements.txt；
我的代码文件在multimodal.py文件内，代码共三部分，为图像分类、文本分类和多模态融合，详见报告report.pdf；
由于文本分类时无法下载bert预训练模型，故我把所需文件手动下载了下来放在bert-based-uncased文件夹内；
执行代码仅需运行这一个文件就行了；
参考的库有numpy、torch、efficientnet_pytorch、torchvision和transformers；
测试集结果文件为test.txt。