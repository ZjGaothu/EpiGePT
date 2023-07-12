#!/bin/bash
#SBATCH -G 4
#SBATCH --mem=250G
#SBATCH -p whwong
#SBATCH -n 32
#SBATCH --time=7-00:00
#SBATCH -e error_pre.txt
#SBATCH -o output_pre.txt

#ml python/3.6.1
#ml load cuda/11.2.0
#ml load cudnn/8.1.1.33

#ml load cuda/10.0.130
#ml load cudnn/7.4.1.5

ml load cuda/10.1.105
ml load cudnn/7.6.5

#ml load py-tensorflow/2.4.1_py36
#ml load py-tensorflow/2.1.0_py36
#ml biology
#ml bedtools/2.27.1
#ml samtools/1.8
#bash download_raw_data_peaks.sh -i $1 -c -d 

function overlapBin()
{
    #for file in data/encode/dseq_peaks/*.bed
    for file in data/encode/cseq_peaks/*.bed
    do
        output=${file/.bed/.128.overlap.bin}
        echo $file, $output
        bedtools intersect -wa -a data/encode/hg19.128.bed -b $file |uniq  > $output
    done
}

function getReadsCount()
{
    input_bam=$1
    len=$2
    fbed=${input_bam/bam/`echo $len`.bed}
    samtools index $input_bam
    bedtools multicov -bams $input_bam -bed data/encode/selected.$len.bin > $fbed
}

#overlapBin
#cat data/encode/bam_files.txt |while read line; getReadsCount $line 128k; done
    
#wc -l data/encode/cseq_peaks/*bin data/encode/dseq_peaks/*bin |awk '{print $2}'|grep overlap.bin|xargs cat | sort | uniq -c |awk '$1>9&&$2!="chrY"{print $2"\t"$3"\t"$4}'|bedtools sort -i > union.all.overlap.bin

#macs3 callpeak -t $1 -c $2 -f BAM -g hs -n test -B -q 0.01 --outdir $3

#len=64k
#p=$1
#findMotifsGenome.pl data/motif/selected.$len.bin.homer.bed /home/users/liuqiao/work/hg19.fa data/motif/motifout_${len}_p$p -find data/motif/all_motif_rmdup_p$p.motif -p 32 -cache 250000 > data/motif/motifscan.$len.p$p.txt

#/share/software/user/open/python/3.6.1/bin/python3 preprocess.py 
~/anaconda3/condabin/conda activate geformer
/home/users/liuqiao/anaconda3/envs/geformer/bin/python3.6 main.py
#/share/software/user/open/python/3.6.1/bin/python3 main.py 
