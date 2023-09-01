#!/bin/bash

# 获取系统 CPU 使用率
cpu=$(top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}')

# 获取系统内存使用率
mem=$(free | grep Mem | awk '{print $3/$2 * 100.0}')

# 获取系统磁盘使用率
disk=$(df -h | awk '$NF=="/"{printf "%s", $5}')

# 获取系统负载平均值
load=$(uptime | awk '{print $10 $11 $12}')

# 获取系统当前时间
time=$(date +"%Y-%m-%d %H:%M:%S")

# 输出监控结果
echo "$time CPU: $cpu% Mem: $mem% Disk: $disk Load: $load"

# Add into crontab
# * * * * * /path/to/monitor.sh >> /var/log/monitor.log 2>&1 