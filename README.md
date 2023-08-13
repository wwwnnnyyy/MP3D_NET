# MP3D_NET

**Why do we need MP3D-NET?**

Deep learning (DL) has demonstrated the potential in processing nonlinear mappings from data to inversion models based on previous great efforts. However, DL joint inversion in geophysics has not been made possible for multiple input (including 1D, 2D and 3D) and multiple output models, which are often necessary in complex geological settings and real observations. It is very common that geophysical data obtained in practice often differ in dimension and size. 

So, it is often necessary to perform preprocessing such as "clipping" and "padding" of the data. This means that each batch of input requires complicated preprocessing and some information loss. In order to make the neural network accept geophysical information in multiple modalities, we designed a fully adaptive network and joint inversion framework for the recovery of 3D geological structures using geophysical data, called MP3D-NET (Multi-Physics 3D Net). It was developed from a geophysics application, but our framework has a lot of potential to be applied well in many fields.

# Usage

- model is in model.py
- run python main.py to train, valid and test the model
- notebook latefusion.ipynb is the same as the main.py, which may be an easier way to understand
run the test.py to check whether the model can work properly.

# Environment
```  conda env create -f mp3d.yml  ```
- link to this repository 
- execute this command to set up mp3d environment 
- run all the code in this environment

# Visualization

<img src="https://github.com/wwwnnnyyy/MP3D_NET/assets/61683792/a3720d5a-2e36-42c1-84b0-cf26f23d4d15" width="500" height="340">
<br/><br/><br/>
<img src="https://github.com/wwwnnnyyy/MP3D_NET/assets/61683792/91540836-d19e-41fa-815c-6e2e40573034" width="500" height="340">
<br/><br/><br/>
<img src="https://github.com/wwwnnnyyy/MP3D_NET/assets/61683792/cd7c59be-e7be-4a4f-aa9c-187901e90065" width="500" height="340">
<br/><br/>

- Each geometry of four groups of (a), (b), (c), and (d) tetrahedra was randomly selected in the test set. 
- The three rows of each group show three XoY, XoZ, and YoZ slices.
- The three columns of each group show the slices of the real model, the prediction probability of mineralization, the prediction of the mineralization result given a threshold of 0.1.
