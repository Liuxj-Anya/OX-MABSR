# OX-MABSR

A benchmark for testing how well AI models can understand sentiment in images and text together. Published at AAAI 2026.

## What It Does

You give it an image and some text. It tells you the sentiment — positive, negative, or neutral — and explains why. The benchmark has a dataset, training code, and a web demo so you can try it out.

## Try It

Start the web demo:

```bash
cd frontend
npm install
npm run dev
```

## Run the Models

```bash
pip install torch transformers peft numpy
python CODE/inference.py   # run inference
python CODE/train.py        # train a model
```

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1ayN8fIQF1aLegnNIVM8pFiegRI8BQKMo?usp=sharing).

## Paper

Read the full paper in `paper.pdf`.

## Citation

```bibtex
@inproceedings{liu2026oxmabsr,
  title={OX-MABSR: A Benchmark for Open-domain Explainable Multimodal Aspect-Based Sentiment Reasoning},
  author={Liu, X.J. and Anya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```
