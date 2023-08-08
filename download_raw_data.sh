#!/bin/bash
export LC_ALL=C
PPATH=`readlink -f $BASH_SOURCE | sed -r 's/^\.\///g'`
PPATH=`dirname $PPATH`
DPATH="$PPATH/data/encode"
declare -A GNMAP
GNMAP=([hg19]=hg19 [hg38]=GRCh38)
gname=${GNMAP['hg19']}
#meta data from encode
mpath="$DPATH/metadata.tsv"

function printHelp()
{
    echo "Usage: bash $(basename "$0") [-i CELLID] [-h] [-r] [-c] [-d] "
    echo "-- a program to download RNA-seq data(.tsv), ChIP-seq data(.bam), and DNase-seq data(.bam) from the Encode project (https://www.encodeproject.org)"
    echo "OPTION:"
    echo "    -i  CELLID: pre-defined cell ID (from 1 to 39)"
    echo "    -h  show this help text"
    echo "    -r  download RNA-seq data (.tsv)"
    echo "    -c  download ChIP-seq data (.bam)"
    echo "    -d  download DNase-seq data (.bam)"

}



#download RNA-seq data (.tsv file)
function downloadRseq()
{
    count=0
    cat  $mpath		|
    tail -n +2			|
    grep -P "$gname"    |
    awk -F '\t' '$8=="polyA plus RNA-seq" || $8=="total RNA-seq"' |
    awk -F '\t' '$5=="gene quantifications"' |
    grep -P "tsv"		|
    grep -P "$CELL"     |
    awk  -F "\t"   	'
    {
        print $1 "\t" $48 "\t" $46 "\t" $7 "\t" $11
    }' |
    while read line; do
        let count+=1
        item=(${line//\\t/})
        facc=${item[0]}
        furl=${item[1]}
        msum=${item[2]}
        exp=${item[3]}
        cellname=${item[4]}
        echo $furl $cellname $CELL
        fbed=$Rseq/$exp/$facc.tsv
        fmd5=$Rseq/$exp/$facc.md5
        `
        if [ $cellname == $CELL ]
        then
          for ((i = 0; i < 5; i++)); do
            if [[ ! -e $fmd5 || \`cat $fmd5\` != $msum ]]; then
                mkdir -p \`dirname $fbed\`
                printf "[Downloading] %s\t%s\t%s\t%s\n" $facc $furl $msum $exp
                curl  -o $fbed -L $furl -C - -s
                if [  -e $fbed ]; then
                    md5sum $fbed | cut -d ' ' -f 1 > $fmd5
                fi
            fi
        done  
        fi
        ` 
#         sleep 1s; while [ `ps -T | grep -P "\s+curl$" | wc -l` -ge 5 ]; do sleep 1s; done
    echo "$count RNA-seq .tsv files download finished"
    done
}


#download ChIP-seq data (.bam file)
function downloadCseq()
{
    #downloading ChIP-seq data(TF+histone)
    count=0
    cat $mpath |
    tail -n +2			|
    grep -P "$gname"	|
    grep ChIP-seq       |
    grep bam            |
    grep hg19 |
    awk -F '\t' '$5=="alignments"' |
    #remove control and histone marker (chip-seq)
    #grep -v -P "control" |
    #grep -v -P "\sH\d+"  |
    grep -P "$CELL"		  |
    grep -F -f "$DPATH/targets.txt" |
    #sort -d	-k1,1	  |
    awk  -F "\t"   	'
    {
        print $1 "\t" $48 "\t" $46 "\t" $7 "\t" $23 "\t" $11
    }' |
    while read line; do
        let count+=1
        item=(${line//\\t/})
        facc=${item[0]}
        furl=${item[1]}
        msum=${item[2]}
        exp=${item[3]}
        target=${item[4]}
        cellname=${item[5]}
        fbed=$Cseq/$target/$exp/$facc.bam
        fmd5=$Cseq/$target/$exp/$facc.md5
        echo $furl >> ./ChIP_target.txt
        `
        for ((i = 0; i < 5; i++)); do
            if [[ ! -e $fmd5 || \`cat $fmd5\` != $msum ]]; then
                mkdir -p \`dirname $fbed\`
                printf "[Downloading] %s\t%s\t%s\t%s\n" $facc $furl $msum $exp
                curl  -o $fbed -L $furl -C - -s
                if [  -e $fbed ]; then
                    md5sum $fbed | cut -d ' ' -f 1 > $fmd5
                fi
            fi
        done
        ` 
            #sleep 1s; while [ `ps -T | grep -P "\s+curl$" | wc -l` -ge 5 ]; do sleep 1s; done
        echo "${count} ChIP-seq .bam files download finished"
        done
    #echo "${count} ChIP-seq .bam files download finished"
}


#download DNase-seq data (.bam file)
function downloadDseq()
{
count=0
cat  $mpath		|
tail -n +2			|
grep -P "$gname"	|
grep DNase-seq |
grep bam |
grep hg19 |
awk -F '\t' '$5=="alignments"' |
grep -P "$CELL"		  |
#sort -d	-k1,1	  |
awk  -F "\t"   	'
{
	print $1 "\t" $48 "\t" $46 "\t" $7 "\t" $11
}' |
while read line; do
        let count+=1
	item=(${line//\\t/})
	facc=${item[0]}
	furl=${item[1]}
	msum=${item[2]}
	exp=${item[3]}
    cellname=${item[4]}
	fbed=$Dseq/$exp/$facc.bam
	fmd5=$Dseq/$exp/$facc.md5
    echo $furl >> DNase_target1.txt
    echo $fbed >> path_target1.txt
    `
    if [$cellname == $CELL]
    then
      for ((i = 0; i < 5; i++)); do
        if [[ ! -e $fmd5 || \`cat $fmd5\` != $msum ]]; then
            mkdir -p \`dirname $fbed\`
            printf "[Downloading] %s\t%s\t%s\t%s\n" $facc $furl $msum $exp
            curl  -o $fbed -L $furl -C - -s
            if [  -e $fbed ]; then
                md5sum $fbed | cut -d ' ' -f 1 > $fmd5
            fi
        fi
    done  
    fi
    ` 
# 	sleep 1s; while [ `ps -T | grep -P "\s+curl$" | wc -l` -ge 5 ]; do sleep 1s; done
echo "${count} DNase-seq .bam files download finished."
done
#echo "${count} DNase-seq .bam files download finished."
}

while getopts "i:hrcd" Option;do
    case $Option in 
        h) printHelp
        exit 1
        ;;
        i) 
            CELLID=$OPTARG
            CELL=$(sed -n ${CELLID}p $DPATH/celltypes_and_tissues.txt|cut -f 2)
            mkdir -p "$DPATH/raw_data/$CELLID"
            Rseq="$DPATH/raw_data/$CELLID/rseq"
            Dseq="$DPATH/raw_data/$CELLID/dseq"
            Cseq="$DPATH/raw_data/$CELLID/cseq"
        ;;
        r) downloadRseq
        ;;
        c) downloadCseq
        ;;
        d) downloadDseq
        ;;
        \?) printHelp 
        exit 1
        ;;
    esac
done
