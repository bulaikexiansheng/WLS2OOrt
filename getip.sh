#!/bin/bash

# 获取网络接口名称
interface=$(ip -o -4 route show to default | awk '{print $5}')

# 获取IP地址
ip_address=$(ip addr show dev $interface | awk '$1 == "inet" {print $2}' | cut -d'/' -f1)

echo "当前IP地址是: $ip_address"
