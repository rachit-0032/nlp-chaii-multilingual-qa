# chaii - Hindi and Tamil Question Answering

<!-- MarkdownTOC -->

1. [Problem Statement](#problem-statement)
1. [Methodology](#methodology)
	1. [Datasets used](#datasets-used)
	1. [Model](#model)
1. [Running the code](#running-the-code)
	1. [Training](#training)
	1. [Inference](#inference)
1. [Authors](#authors)

<!-- /MarkdownTOC -->


<a id="problem-statement"></a>
## Problem Statement

The goal is to predict answers to real questions about Wikipedia articles using chaii-1 dataset.

chaii-1 is a question answering dataset in Hindi and Tamil (without the use of translation). Given a context and question, the goal of question answering is to predict the answer to the question by selecting a span from the context. Consider the following example from the dataset:

<details open>
<summary><b>Example:</b></summary>
<br/>

> **Context**: मानव कंकाल शरीर की आन्तरिक संरचना होती है। यह जन्म के समय 300 हड्डियों से बना होता है और यवाव ु स्था में कुछ हड्डियों के संगलित होने से यह २०६ तक सीमित हो जाती है।[1] तंत्रिका में हड्डियों का द्रव्यमान ३० वर्ष की आयु के लगभग अपने अधिकतम घनत्व पर पहुँचती है। मानव कंकाल को अक्षीय कंकाल और उपांगी कंकाल में विभाजित किया जाता है। अक्षीय कंकाल मेरूदण्ड, पसली पिजर ं और खोपड़ी से मिलकर बना होता है। उपांगी कंकाल अक्षीय कंकाल से जड़ुा हुआ होता है तथा असं मेखला, श्रोणि मेखला और अधः पाद एवं ऊपरी पाद की हड्डियों से मिलकर बना होता है। मानव कंकाल निम्नलिखित छः कार्य करता है: उपजीवन, गति, रक्षण, रुधिर कणिकाओं का निर्माण, आयनों का भंडारण और अतं : स्रावी विनियमन। मानव कंकाल अन्य प्रजातियों के समान लगिैं क द्विरूपता नहीं रखता लेकिन मस्तिष्क, दंत विन्यास, लम्बी हड्डियों और श्रोणियों में आकीरिकी के अनसार ु अल्प अन्तर होता है। सामान्यतः महिला कंकाल के अवयवों उसी तरह के परुुषों की की तलना ु में कुछ मात्रा में छोटे और कम मजबतू होते हैं। अन्य प्राणियों से भिन्न, मानव परुुष का लिगं स्तंभास्थि रहित होता है।[2] सन्दर्भ श्रेणी:कंकाल तंत्र 

> **Question**: जन्म के समय शिशुके शरीर में कितनी हड्डियाँ होती है? 

> **Answer**: 300

</details>

**This work is based on a Kaggle competition: [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering)**


<a id="methodology"></a>
## Methodology

<a id="datasets-used"></a>
### Datasets used
* *official* chaii-1 dataset [[Link](https://www.kaggle.com/c/chaii-hindi-and-tamil-question-answering/data)]
* MLQA/XQuAD for Hindi [[Link](https://www.kaggle.com/rhtsingh/mlqa-hindi-processed)]
* SQuAD translated to Tamil [[Link](https://www.kaggle.com/msafi04/squad-translated-to-tamil-for-chaii)]

**All these datasets can be found in one place: [[here](https://www.kaggle.com/dataset/006ca8d5f7fa611f2c5acc89b75f4cf7e2b0b4279366c2d8a3d6423e14f4fecb)]**

<a id="model"></a>
### Model
We finetuned XLM-RoBERTa for `QuestionAnswering` from [`alon-albalak/xlm-roberta-large-xquad`](https://huggingface.co/alon-albalak/xlm-roberta-large-xquad) checkpoint.

For hyperparameters, look at [`args`](https://github.com/subhalingamd/nlp-chaii-multilingual-qa/blob/20823005a9a7f7452d1cd2cfab874504e852c28d/train.py#L34:L64) in [`train.py`](train.py).


<a id="running-the-code"></a>
## Running the code
<a id="training"></a>
### Training
To train yourself, use the following command:
```bash
python train.py
```

**While running the training script, the files mentioned in [Section 2.i](#datasets-used) have to be placed under `dataset/` directory (relative to the training script).**

<a id="inference"></a>
### Inference
For inference you can use [`inference.ipynb`](inference.ipynb). 

**You can either fine-tune your own version of the model *(by following the steps mentioned in the previous section)* or use our version which is available [[here](https://www.kaggle.com/dataset/d6b465f7d274361446ce0129469f4e070f235cf8b639b25d424186d1381bc691)].**


<a id="authors"></a>
## Authors
- Subhalingam D [(subhalingamd)](https://github.com/subhalingamd)
- Rachit Jain [(rachit-0032)](https://github.com/rachit-0032)



----
*This README uses texts from the assignment problem document provided in the course.*