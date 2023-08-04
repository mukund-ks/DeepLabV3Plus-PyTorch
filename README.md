<br/>
<p align="center">
  <h3 align="center">DeepLabV3Plus-PyTorch</h3>

  <p align="center">
    A DeepLab V3+ Model with ResNet 50 Encoder to perform Binary Segmentation Tasks. 
    <br/>
    <br/>
    <a href="https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues">Report Bug</a>
    .
    <a href="https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/mukund-ks/DeepLabV3Plus-PyTorch/total) ![Contributors](https://img.shields.io/github/contributors/mukund-ks/DeepLabV3Plus-PyTorch?color=dark-green) ![Forks](https://img.shields.io/github/forks/mukund-ks/DeepLabV3Plus-PyTorch?style=social) ![Stargazers](https://img.shields.io/github/stars/mukund-ks/DeepLabV3Plus-PyTorch?style=social) ![Issues](https://img.shields.io/github/issues/mukund-ks/DeepLabV3Plus-PyTorch) ![License](https://img.shields.io/github/license/mukund-ks/DeepLabV3Plus-PyTorch) 

## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
  - [Creating A Pull Request](#creating-a-pull-request)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)

## About The Project

The goal of this research is to develop a DeepLabV3+ model with a ResNet50 backbone to perform binary segmentation on plant image datasets. Based on the presence or absence of a certain object or characteristic, binary segmentation entails splitting an image into discrete subgroups known as image segments which helps to simplify processing or analysis of the image by reducing the complexity of the image. Labeling pixels is a step in the segmentation process. Each pixel or piece of a picture assigned to the same category has a unique label. 


Plant pictures with ground truth binary mask labels make up the training and validation dataset. The project uses PyTorch, a well-known deep learning library, for model development, training, and evaluation. During the training process, the model is optimized using strategies like the Dice Loss, Adam optimizer, Reducing LR on Pleateau and Early Stopping. All the while, important metrics like Intersection over Union (IoU), Pixel Accuracy, and Dice Coefficient are kept track of.

_A list of datasets used for this project is given below:_
* [EWS Dataset](https://www.research-collection.ethz.ch/handle/20.500.11850/512332)

* [Plant Semantic Segmentation Dataset by HIL](https://humansintheloop.org/resources/datasets/plant-segmentation/)

* [CVPPP Dataset](https://www.plant-phenotyping.org/datasets-home)

The ultimate objective of the project is to develop a strong model that can accurately segment plant-related regions inside photographs, which can have applications in a variety of fields, such as agriculture, botany, and environmental sciences. The included code demonstrates how to prepare the data, create the model's architecture, train it on the dataset, and assess the model's effectiveness using a variety of metrics.

## Built With
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)


* IDE Used:


![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)

* Operating System:

![Windows11](https://img.shields.io/badge/Windows_11-0078d4?style=for-the-badge&logo=windows-11&logoColor=white)

## Getting Started

To get a local copy of this project up and running on your machine, follow these simple steps.

* Clone a copy of this Repository on your machine.
```console
git clone https://github.com/mukund-ks/DeepLabV3Plus-PyTorch.git
```

### Prerequisites

You'll need to have a machine with python 3.9 or greater installed.

```console
python -V
Python 3.9.13
```

### Installation

1. Move into the cloned repo.
```console
cd DeepLabV3Plus-PyTorch
```

2. Setup a Virutal Environment

```console
python -m venv env
```

3. Activate the Virutal Environment
```console
env/Scripts/activate
```

4. Install Dependencies

```console
pip install -r requirements.txt
```

> **Note**
> You can deactivate the Virtual Environment by using
> ```env/Scripts/deactivate```
 

## Usage

You can train on the data aforementioned in the [**About**](#about-the-project) section or use your own data.

* To train the model, use [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py)
```console
python train.py --help
```
```console
Usage: train.py [OPTIONS]

  Training Script for DeepLabV3+ with ResNet50 Encoder for Binary
  Segmentation.

  Please make sure your data is structured according to the folder structure
  specified in the Github Repository.

  See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

  Refer to the Options below for usage.

Options:
  -d, --data-dir TEXT       Path for Data Directory  [required]
  -e, --num-epochs INTEGER  Number of epochs to train the model for. Default -
                            25
  -b, --batch-size INTEGER  Batch size of data for training. Default - 4
  -p, --pre-split BOOLEAN   Opt-in to split data into Training and Validaton
                            set. Default - False
  -a, --augment BOOLEAN     Opt-in to apply augmentations to training set.
                            Default - True
  --help                    Show this message and exit.
```

* For Evaluation, use [`evaluation.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/evaluation.py)
```console
python evaluation.py --help
```
```console
Usage: evaluation.py [OPTIONS]

  Evaluation Script for DeepLabV3+ with ResNet50 Encoder for Binary
  Segmentation.

  Please make sure your evaluation data is structured according to the folder
  structure specified in the Github Repository.

  See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

  Refer to the Option(s) below for usage.

Options:
  -d, --data-dir TEXT  Path for Data Directory  [required]
  --help               Show this message and exit.
```

## Folder Structure

The folder structure will alter slightly depending on whether or not your training data has already been divided into a training and testing set.

- If the data is not already seperated, it should be in a directory called `data` that is further subdivided into `Image` and `Mask` subdirectories.
  
  - [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py) should be run with `--pre-split` option as `False` in this case.
  
    Example: ```python train.py --data-dir data --pre-split False```

> **Note**
> [`dataset.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/dataset.py) file will split the data into training and testing set with a ratio of 0.2

```console
$ tree -L 2
.
├── data
│   ├── Image
│   └── Mask
└── eval_data
    ├── Image
    └── Mask
```

- If the data has already been separated, it should be in a directory called `data` that is further subdivided into the subdirectories `Train` and `Test`, both of which contain the subdirectories `Image` and `Mask`.

  - [`train.py`](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/train.py) should be run with `--pre-split` option as `True` in this case.
  
    Example: ```python train.py --data-dir data --pre-split True```

```console
$ tree -L 3
.
├── data
│   ├── Test
│   │   ├── Image
│   │   └── Mask
│   └── Train
│       ├── Image
│       └── Mask
└── eval_data
    ├── Image
    └── Mask
```
* The structure of `eval_data` remains the same in both cases, holding `Image` and `Mask` sub-directories.
## Roadmap

See the [open issues](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b MyBranch`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push -u origin myBranch`)
5. Open a Pull Request

## License

Distributed under the Apache 2.0 License. See [LICENSE](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch/blob/main/LICENSE) for more information.

## Authors

* [**Mukund Kumar Surehli**](https://github.com/mukund-ks/) - *Comp Sci Student* - *Built the Project.*

## Acknowledgements

* M. Minervini, A. Fischbach, H.Scharr, and S.A. Tsaftaris. [_Finely-grained annotated datasets for image-based plant phenotyping._](https://www.sciencedirect.com/science/article/abs/pii/S0167865515003645?via%3Dihub) Pattern Recognition Letters, pages 1-10, 2015, [doi:10.1016/j.patrec.2015.10.013](https://www.sciencedirect.com/science/article/abs/pii/S0167865515003645?via%3Dihub)
* H. Scharr, M. Minervini, A.P. French, C. Klukas, D. Kramer, Xiaoming Liu, I. Luengo, J.-M. Pape, G. Polder, D. Vukadinovic, Xi Yin, and S.A. Tsaftaris. [_Leaf segmentation in plant phenotyping: A collation study._](https://link.springer.com/article/10.1007/s00138-015-0737-3) Machine Vision and Applications, pages 1-18, 2015, [doi:10.1007/s00138-015-0737-3.](https://link.springer.com/article/10.1007/s00138-015-0737-3)
* B. Dellen, H. Scharr, and C. Torras. [_Growth signatures of rosette plants from time-lapse video._](https://ieeexplore.ieee.org/document/7044561) IEEE/ACM Transactions on Computational Biology and Bioinformatics, PP(99):1 - 11, 2015, [doi:10.1109/TCBB.2015.2404810](https://ieeexplore.ieee.org/document/7044561)
* E.E. Aksoy, A. Abramov, F. Wörgötter, H. Scharr, A. Fischbach, and B. Dellen. [_Modeling leaf growth of rosette plants using infrared stereo image sequences._](https://www.sciencedirect.com/science/article/abs/pii/S0168169914002816?via%3Dihub) Computers and Electronics in Agriculture, 110:78 - 90, 2015, [doi:10.1016/j.compag.2014.10.020](https://www.sciencedirect.com/science/article/abs/pii/S0168169914002816?via%3Dihub)
* M. Minervini , M.M. Abdelsamea, S.A. Tsaftaris. [_Image-based plant phenotyping with incremental learning and active contours._](https://www.sciencedirect.com/science/article/abs/pii/S1574954113000691?via%3Dihub) Ecological Informatics 23, 35–48, 2014, [doi:10.1016/j.ecoinf.2013.07.004](https://www.sciencedirect.com/science/article/abs/pii/S1574954113000691?via%3Dihub)
* Polat H. [_A modified DeepLabV3+ based semantic segmentation of chest computed tomography images for COVID-19 lung infections._](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349869/) Int J Imaging Syst Technol. 2022;32(5):1481-1495. [doi:10.1002/ima.22772](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349869/)
* Li, K. (2022). [_Study on the segmentation method of the improved deeplabv3+ algorithm in the basketball scene._](https://www.hindawi.com/journals/sp/2022/3311931/) Scientific Programming, 2022, 1–7. https://doi.org/10.1155/2022/3311931
* Wang Y, Wang C, Wu H, Chen P (2022) [_An improved Deeplabv3+ semantic segmentation algorithm with multiple loss constraints._](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261582) PLOS ONE 17(1): e0261582. https://doi.org/10.1371/journal.pone.0261582
* Zenkl, R., Timofte, R., Kirchgessner, N., Roth, L., Hund, A., Van Gool, L., Walter, A., &amp; Aasen, H. (2022). [_Outdoor plant segmentation with deep learning for high-throughput field phenotyping on a diverse wheat dataset._](https://www.frontiersin.org/articles/10.3389/fpls.2021.774068/full) Frontiers in Plant Science, 12. https://doi.org/10.3389/fpls.2021.774068 
* Hsu C-Y, Hu R, Xiang Y, Long X, Li Z. [_Improving the Deeplabv3+ Model with Attention Mechanisms Applied to Eye Detection and Segmentation._](https://www.mdpi.com/2227-7390/10/15/2597) Mathematics. 2022; 10(15):2597. https://doi.org/10.3390/math10152597
* Singh, V. (2023, January 17). _The Ultimate Guide to deeplabv3 - with Pytorch Inference._ LearnOpenCV. https://learnopencv.com/deeplabv3-ultimate-guide/ 
* 
* [EWS Dataset](https://doi.org/10.3389/fpls.2021.774068)
* [Plant Phenotyping Dataset](https://www.plant-phenotyping.org/datasets)
* [Plant Semantic Segmentation Dataset by HIL](https://humansintheloop.org/resources/datasets/plant-segmentation/)
