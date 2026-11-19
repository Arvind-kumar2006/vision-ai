"""CATR (CAption TRansformer) model code.

ResNet-101 (dilated) encoder + 6-layer transformer encoder/decoder, BERT
word-piece vocabulary, trained on COCO 2017.
Served by app/services/caption_service.py when `inference.model: catr`.
"""
