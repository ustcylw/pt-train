

DEBUG=true
mode='norm'
time_now=`date "+%Y:%m:%d %H:%M:%S"`



if [ "$1" = "$mode" ];then
    echo $1
    DEBUG=false
fi
echo "debug="$DEBUG



if $DEBUG;then
# python src/train/ddp_train_v3.py
python src/train/ddp_train_v3_1.py
else
# train and test
# CONFIGS.MODE = 'train' / 'test'
# python src/train/train_v1.py
# nohup python -u src/train/ddp_train_v3.py > /data/ylw/code/pl_yolo_v5/runs/"$time_now".log 2>1& &
# nohup python src/train/ddp_train_v3.py > /data/ylw/code/pl_yolo_v5/runs/"$time_now".log 2>&1 &


# nohup python src/train/ddp_train_v3.py > /data/ylw/code/pl_yolo_v5/runs/"$time_now".log &
nohup python src/train/ddp_train_v3_1.py > /data/ylw/code/pl_yolo_v5/runs/"$time_now".log &
fi




# nohup python -u train.py --use-cuda --iters 200 --dataset coco --data-dir /home/leinao/data/coco2017 &
