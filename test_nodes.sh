I=(1 2 4 8)
J=(1 2 4 8)

for i in ${I[@]}; do
    for j in ${J[@]}; do
        if [ $i -lt $j ]; then
            result_file="record/nodes.txt"
            echo "运行程序，结果保存到 $result_file"
            output=$(bash test.sh ann $i $j)
            # 提取召回率和延迟时间
            recall=$(echo "$output" | grep "平均召回率")
            latency=$(echo "$output" | grep "平均查询延迟 (微秒):")
            echo "$i,$j,$recall,$latency" >> node.csv
        fi
    done
done


