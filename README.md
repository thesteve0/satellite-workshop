# satellite-workshop

Decisions I need to make
1. Jupyter
2. Segmentation set


### Steps so far - get these working in code
1. We are going to do predictions on this data with this model
   2. caltech101_dataset = foz.load_zoo_dataset("caltech101")
   3. modelresx = foz.load_zoo_model("resnext50-32x4d-imagenet-torch")
4. Then we are going to run that same model on the Dubai but we need to chip the data first
   https://huggingface.co/docs/transformers/tasks/image_classification
5. Then do embeddings on the chips\\

5. Then do grounding object detection on dubai data
   https://huggingface.co/docs/transformers/tasks/zero_shot_object_detection
6. 
Then do embeddings on our objects
Should we then do semantic seg on dubai or is not the time for multispectral?

https://huggingface.co/docs/transformers/tasks/semantic_segmentation#types-of-segmentation

