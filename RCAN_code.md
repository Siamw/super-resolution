# RCAN code

[github](https://github.com/yulunzhang/RCAN)

## Dataset prepare

1. Dataset Download (위 링크 참조) - data양이 많아 꽤 오래걸림
2. put dataset at RCAN_TrainCode/code/traindata - code밖에 만들어도, 다른이름으로 만들어도 가능. 아래 path만 잘 설정하면됨
3. edit RCAN_TrainCode/option.py       
    - parser.add_argument('--dir_data', type=str, default='../traindata',help='dataset directory')
4. and unzip. 
    * unzip at Colab : !unzip -uq "../traindata/DIV2K/DIV2K_valid_LR_bicubic_X4.zip" -d "../traindata/DIV2K"

## Train
1. !python3 main.py --model RCAN --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96
        -> FileNotFoundError: [Errno 2] No such file or directory: '../traindata/DIV2K/DIV2K_train_HR/0801.png'


data가 800까지밖에 없는데 더 불러오려고 해서 에러남. 어느 코드에서 dataset 코드가 있는지 알아봐야할듯.
