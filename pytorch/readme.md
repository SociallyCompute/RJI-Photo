# READ Me for installing

1. create a virtual environment
2. `pip install torch torchvision`
3. To use TorchVision for object detection, which is useful for the RJI project because we theorize that the "shape of objects in a photo" will be strong signals of the "photo category": https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
	- pip install Cython
    - pip install pycocotools
    - From Here: https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=DBIoe_tHTQgV
        - git clone https://github.com/cocodataset/cocoapi.git
        - cd cocoapi/PythonAPI
        - python setup.py build_ext install
4.  