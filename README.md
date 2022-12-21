# Quantify nuclei (cells)

A project to identifying and quantifying the DNA damage cell
    
## Prerequisites

- [Python](https://www.python.org/downloads/) - v3.10.8
- [cv2](https://pypi.org/project/opencv-python/) - 4.6.0
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html) - 3.6.2
- [pandas](https://pypi.org/project/pandas/) - 1.5.2
- [skimage](https://pypi.org/project/scikit-image/) - 0.19.3

## Setup
Clone the repository, install the dependencies and get started right away.

    $ git clone git@github.com:aishstha/Quantify-nuclei.git
    $ cd Quantify-nuclei
    $ ./segment_script.py <Input-image-file-path> 
    
    
    Example : 
    
      ```bash
      ./segment_script.py  "Data/IXMtest_C05_s7_w1F71963FB-8F29-41CB-A5F5-07CB9584BBC5.tif"

      ```
## Usage

1: You can manually Install packages mentioned above (cv2, numpy, matplotlib, pandas, sys, skimage)
OR 
2: Use the conda environment file for this project. 
    $ Check `environment.yml` 
    
## Creating an environment from an environment.yml file (Conda Environment)



## Output

Check `data_to_excel.xlsx` file for stats and `output.jpg` for visual result. The output.jpg file has the labeling in the cells, the information about each cell is according to the label in excel file. 
