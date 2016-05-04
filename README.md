# COS495-project
To run network, create three files that contains (a) training images and (b) validation images, in a .txt of the form 

```
filepathToImage label
```

The run 

```
python statefarm_train.py <train.txt> <validate.txt>
```

You must ensure the dimensions of your training and validation images match each other and the image dimensions on line 30 of statefarm_input.py

```
IMAGE_SIZE = (75, 100)
```

To make predictions on a set of (unlabeled) testing data, create a .txt containing one filepath per line. Then run

```
python statefarm_eval.py <test.txt>
```