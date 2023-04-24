# MM-PCQA 🚀🚀🚀
Official repo for 'MM-PCQA: Multi-Modal Learning for No-reference Point Cloud Quality Assessment', which has been accepted to IJCAI2023.

## Motivation

<img src="https://github.com/zzc-1998/MM-PCQA/blob/main/pics/spotlight.png" align="left" />

Examples of reflected distortions. The point clouds can explicitly reveal the geometry down-sampling distortion while failing to recognize texture noise unless the projections are involved, which raises the need for multi-modal perception. Therefore, to boost the performance of PCQA, we propose a multi-modal learning strategy for NR-PCQA, which extracts quality-aware features not only from the 3D point clouds but also from the 2D projections. 
There are two main reasons to adopt this strategy. First, point clouds can be perceived in both 2D/3D scenarios. We can view point clouds from 2D perspective via projecting them on the screen or directly watch point clouds in 3D format using the VR equipment. Thus multi-modal learning is able to cover more range of practical situations. Second, different types of distortions have divergent visual influence on different modalities. As shown in figure below, the structural damage and geometry down-sampling are more obvious in the point cloud modality while the image modality is more sensitive to texture distortions caused by color quantization and color noise. Moreover, it is easier to extract quality-aware semantic information from the image modality. Thus, the proposed multi-modal learning fashion can make up for the deficiencies and take advantage of both modalities.





## Framework

<p align="center">
  <img src="https://github.com/zzc-1998/MM-PCQA/blob/main/pics/framework.png" /> 
</p>

The framework is exhibited above. Considering that the local patterns such as smoothness and roughness are very important for quality assessment, we first propose to split the point cloud into sub-models rather than sampling points for analysis, which has been previously adopted for extracting the low-level pattern features of the point cloud. The image projections are acquired by rendering the point clouds from several viewpoints with a fixed viewing distance to maintain the consistency of the texture scale. Then a point cloud encoder and an image encoder are used to encode the point clouds and projected images into quality-aware embeddings respectively, which are subsequently strengthened by symmetric cross-modality attention. Finally, the quality-aware embeddings are decoded into quality values with fully-connected layers.

# How to run the code 🌟🌟🌟

## Environment Build

We train and test the code on the Ubuntu 18.04 platform with open3d=0.15 and pytorch=1.8. The GPU is 3090 with 24 GB memory, the batchsize 8 is fine.

## Begin training

You can simply train the MM-PCQA by referring to train.sh. For example, you can train MM-PCQA on the SJTU-PCQA database with the following command:

```
CUDA_VISIBLE_DEVICES=0 python -u train.py \
--learning_rate 0.00005 \
--model MM_PCQA \
--batch_size  8 \
--database SJTU  \
--data_dir_2d path_to_sjtu_projections \
--data_dir_pc path_to_sjtu_patch_2048 \
--loss l2rank \
--num_epochs 50 \
--k_fold_num 9 \
>> logs/sjtu_mmpcqa.log
```

You only need to replace the path of 'data_dir_2d' and 'data_dir_pc' with the path of data on your computer. **We provide the download links of the projections and patches, which can be accessed here ([Onedrive](https://1drv.ms/f/s!AjaDoj_-yWggygWzjplEICwa2G9k?e=5x7b8i) [BaiduYunpan](https://pan.baidu.com/s/1SuDsQxSRGJ5jePjhTPatHQ?pwd=pcqa)).**  

By unzipping the files, you should get the file structure like:

```
├── sjtu_projections
│   ├── hhi_0.ply
│   │   ├── 0.png
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
...
├── sjtu_patch_2048
│   ├── hhi_0.npy
│   ├── hhi_1.npy
│   ├── hhi_2.npy
...
```

Then change the path of 'data_dir_2d' and 'data_dir_pc' to 'path.../sjtu_projections' and 'path.../sjtu_patch_2048'. 

If you want to generate the patches and projections by your self, you can simply refer to 'utils/get_patch.py' and 'utils/get_projections.py' for help.

## Test

We also provide the 'test_single_ply.py' to quikly test the MM-PCQA on your own point clouds. 

```
parser.add_argument('--objname', type=str, default='bag/bag_level_7.ply') # path to the test ply
parser.add_argument('--ckpt_path', type=str, default='WPC.pth') # path to the pretrained weights
```
You only need to give the path to your point clouds (.ply) and pretrained MM-PCQA weights.

**We provide the weights trained on the WPC database here (WPC.pth [Onedrive](https://1drv.ms/f/s!AjaDoj_-yWggygWzjplEICwa2G9k?e=5x7b8i) [BaiduYunpan](https://pan.baidu.com/s/1SuDsQxSRGJ5jePjhTPatHQ?pwd=pcqa))**.

Then you can get the visual scores of the point clouds. During our test, the total computation time per point colud is within 10 seconds.

# Citation ☕☕☕ 

If you find our work useful, please star our project and cite our paper as:

```
@article{zhang2022mm,
  title={MM-PCQA: Multi-Modal Learning for No-reference Point Cloud Quality Assessment},
  author={Zhang, Zicheng and Sun, Wei and Min, Xiongkuo and Zhou, Quan and He, Jun and Wang, Qiyuan and Zhai, Guangtao},
  journal={IJCAI},
  year={2023}
}
```
