# GAN_flownet

This repository uses Generative Adversarial Networks to improve [Steady-State-Flow-With-Neural-Nets](https://github.com/loliverhennigh/Steady-State-Flow-With-Neural-Nets)

# Dataset
You can get the train dataset from [here](https://drive.google.com/file/d/0BzsbU65NgrSuZDBMOW93OWpsMHM/view?usp=sharing) and put it to `data` directory. You can get the test dataset from [here](https://drive.google.com/file/d/0BzsbU65NgrSuR2NRRjBRMDVHaDQ/view?usp=sharing) . Unzip it and put it to `data` directory. 
# Train
You can run:
```Shell
python train/flow_train.py
```

# Test 
You can run:
```Shell
python test/flow_test.py
```

# Tensorboard
[Loss_dis](https://github.com/jiazhaozhu/GAN_flownet/blob/master/checkpoints/Loss_dis.jpg)
[Loss_ge](https://github.com/jiazhaozhu/GAN_flownet/blob/master/checkpoints/Loss_ge.jpg)
[Loss_squ_ge](https://github.com/jiazhaozhu/GAN_flownet/blob/master/checkpoints/Loss_squ_ge.jpg)
[Loss_cro_ge](https://github.com/jiazhaozhu/GAN_flownet/blob/master/checkpoints/Loss_cro_ge.jpg)

# Debug
You can debug:
`python train/flow_train.py --debug True`
You can use tensorboard to debug:
`python train/flow_train.py --tensorboard_debug True --tensorboard_debug_address 10.11.11.11:6064`

![alt tag](https://github.com/jiazhaozhu/GAN_flownet/blob/master/checkpoints/tensorboard_debug.jpg)

# Results

![alt tag](https://github.com/jiazhaozhu/GAN_flownet/blob/master/test/sample_1.png)
![alt tag](https://github.com/jiazhaozhu/GAN_flownet/blob/master/test/sample_2.png)
![alt tag](https://github.com/jiazhaozhu/GAN_flownet/blob/master/test/sample_3.png)
