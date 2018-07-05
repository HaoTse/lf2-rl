# LF2 with reinforcement learning

## Requirements

### Enviornment
- ubuntu 16.04
- python 3.6.5
- pytroch 0.4.0
- [lf2gym](https://github.com/HaoTse/lf2gym)

### Prerequisite
- install package
```shell
pip install -r requirements.txt
```

- install lf2gym
```
bash setup.sh
```
> If you receive a premissions error, you should use `chmod` command to change the premission of the diver you use in `lf2gym/webdriver/*`.

## Model

1. CNN

![image](https://github.com/HaoTse/lf2-rl/blob/master/img/cnn.png)

2. Concat features to Linear layer

![image](https://github.com/HaoTse/lf2-rl/blob/master/img/linear.png)

## Result
### Davis

- Picture mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_davis_picture_test.gif)

- Feature mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_davis_feature_test.gif)

- Mix mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_davis_mix_test.gif)

### Firen

- Picture mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_firen_picture_test.gif)

- Feature mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_firen_feature_test.gif)

- Mix mode
![image](https://github.com/HaoTse/lf2-rl/blob/master/gif/new_firen_mix_test.gif)

### learning curve

- steps

![image](https://github.com/HaoTse/lf2-rl/blob/master/img/step.png)

- rewards

![image](https://github.com/HaoTse/lf2-rl/blob/master/img/reward.png)

## Analysis
> See `report.pdf`.
