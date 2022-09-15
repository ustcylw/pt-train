
tmux new -s test_tmux

time_now=`date "+%Y:%m:%d %H:%M:%S"`

nohup ping baidu.com > ./logs/"$time_now".log 2>&1 &
