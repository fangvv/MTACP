#!/bin/bash
#遍历文件夹、删除指定后缀名的文件
#问题：受保护的文件删除不了
 
function scandir() {
    local cur_dir parent_dir workdir
    workdir=$1
    cd ${workdir}
    if [ ${workdir} = "/" ]
    then
        cur_dir=""
    else
        cur_dir=$(pwd)
    fi
 
    for dirlist in $(ls ${cur_dir})
    do
        if test -d ${dirlist};then
            cd ${dirlist}
            scandir ${cur_dir}/${dirlist} $2
            cd ..
        else
            #echo ${cur_dir}/${dirlist}
            #做自己的工作
            local filename=$dirlist
            #echo "当前文件是："$filename
            #echo ${#2} #.zip 4
            #echo ${filename:(-${#2})}

        	if [[ ${filename:(-${#2})} = $2 ]] 
        	then
        		echo "删除文件"$filename
 				rm -f $filename
 			fi
        fi
    done
}
 
if test -d $1
then
    scandir $1 $2
elif test -f $1
then
    echo "you input a file but not a directory,pls reinput and try again"
    exit 1
else
    echo "the Directory isn't exist which you input,pls input a new one!!"
    exit 1
fi

## 用法:sh clean_log.sh ./test_dir/ .zip 遍历文件夹test_dir及其子文件夹下,找到所有后缀名为.zip的文件,删除它们.
