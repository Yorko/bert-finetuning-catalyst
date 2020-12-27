# BERT classifier fine-tuning with PyTorch, HuggingFace, and Catalyst

<img src='https://habrastorage.org/webt/ne/n_/ow/nen_ow49hxu8zrkgolq1rv3xkhi.png'>

**Instruction:**

- specify your data, model, and training parameters in `config.yml`
- if needed, customize the code for data processing in `src/data.py`
- specify your model in `src/model.py`, by default it's DistilBERT for sequence classification
- run `python src/train.py`

**Video-tutorial**

I explain the pipeline in detail in a video-tutorial which consists of 4 parts:

- [Intro](https://youtu.be/fPDUcaLPu58): overview of this pipeline
- [Data preparation](https://youtu.be/7zoNJV67dkA) for training: from CSV files to PyTorch DataLoaders
- [The model](https://youtu.be/StbFK_rp_rY): understanding the BERT classifier model by HuggingFace, digging into the code of the transformers library
- [Training](https://youtu.be/A5brO93bRis): running the pipeline with Catalyst and GPUs

Also, see other tutorials/talks on the topic:

 - multi-class classification: classifying Amazon product reviews into categories, [Kaggle Notebook](https://www.kaggle.com/kashnitsky/distillbert-catalyst-amazon-product-reviews)
 - multi-label classification: identifying toxic comments, [Kaggle Notebook](https://www.kaggle.com/kashnitsky/catalyst-distilbert-multilabel-clf-toxic-comments)
 - an overview of this pipeline is given in a video [Firing a cannon at sparrows: BERT vs. logreg](https://youtu.be/JIU6WZuWl6k)
