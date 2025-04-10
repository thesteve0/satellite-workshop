import fiftyone as fo
import fiftyone.zoo as foz

ds = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(ds)

session.wait()
