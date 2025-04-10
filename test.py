import fiftyone as fo
import fiftyone.zoo as foz

# caltech101_dataset = foz.load_zoo_dataset(
#     "caltech101",
#     persistent=True,
#     overwrite=True,
#
# )

dataset = foz.load_zoo_dataset(
    "caltech256",
    max_samples=100,
    shuffle=True
)

# dataset = foz.load_zoo_dataset(
#     "coco-2017",
#     split="validation",
#     max_samples=200,
#     shuffle=True,
#     num_workers=6
# )

# dataset = foz.load_zoo_dataset(
#     "open-images-v7",
#     label_types="classifications",
#     split="validation",
#     max_samples=200,
#     shuffle=True,
#     num_workers=8
# )


# dataset3 = foz.load_zoo_dataset(
#     "places",
#     split="validation",
#     max_samples=200,
#     shuffle=True,
# )

# print("Loading model")
modelresx = foz.load_zoo_model("resnext50-32x4d-imagenet-torch")
# # print("running model")
dataset.apply_model(modelresx, label_field="predictions")
# print("done with predictions")

# ds = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset)

session.wait()
