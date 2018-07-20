while true
do
    procnum=`ps -ef|grep "python main.py -m Town" | grep -v grep | wc -l`
    if [ $procnum -eq 0 ]
    then
	export CUDA_VISIBLE_DEVICES=0
        nohup python main.py -m Town01 -l&
        echo `date +%Y-%m-%d` `date +%H:%M:%S`  "restart town01" >> ~/shell.log
    fi
    sleep 1
done
