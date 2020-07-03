# Road-Surface-Classification
In emerging countries it’s common to find unpaved roads or roads with no maintenance. Unpaved or damaged roads also impact in higher fuel costs and vehicle maintenance. This kind of analysis can be useful for both road maintenance departments as well as for autonomous vehicle navigation systems to verify potential critical points. The road type and quality classifier was done through a simple Convolutional Neural Network with few steps.

[![Road Surface Classification](https://i.imgur.com/wQU0CHa.png)](https://youtu.be/3UM97O0MQ3w "Road Surface Classification")

 ## Citation:
 ```
@article{rtk:2019,
  author = {Thiago Rateke and Karla Aparecida Justen and Aldo von Wangenheim},
  title = {Road Surface Classification with Images Captured From Low-cost Cameras – Road Traversing Knowledge (RTK) Dataset},
  journal = {Revista de Informática Teórica e Aplicada (RITA)},
   volume = {26},
  year = {2019},
   doi = {https://doi.org/10.22456/2175-2745.91522},
}
```

## Datasets
For the models trained here, 3 datasets were used:
 - RTK dataset [1].
 - KITTI dataset [2].
 - CaRINA dataset [3].

To read more about the RTK dataset and access it, please visit [http://www.lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en](http://www.lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en).

In this work we used the CNN structure presented by [4] with few adaptations to our problem, such as: sizes, Region of Interest, and data augmentation.
 
 ## REFERENCES:
- [1] - Rateke T, Justen KA, von Wangenheim A. Road surface classification with images captured from low-cost cameras – Road Traversing Knowledge (RTK) dataset. Revista de Informática Teórica e Aplicada (RITA). Url: http://www.lapix.ufsc.br/pesquisas/projeto-veiculo-autonomo/datasets/?lang=en </br>
- [2] - GEIGER, A. et al. Vision meets robotics: The kittidataset. Int. J. Rob. Res., Sage Publications, Inc., ThousandOaks, CA, USA, v. 32, n. 11, p. 1231–1237, sep 2013. Doi:hhttp://dx.doi.org/10.1177/0278364913491297i. Url: http://www.cvlibs.net/datasets/kitti/raw_data.php </br>
- [3] - SHINZATO, P. Y. et al. Carina dataset: An emerging-country urban scenario benchmark for road detection systems.In:2016 IEEE 19th International Conference on IntelligentTransportation Systems (ITSC). [S.l.:  s.n.], 2016. p. 41–46.Doi:hhttp://dx.doi.org/10.1109/ITSC.2016.7795529i. Url: http://www.lrm.icmc.usp.br/dataset  </br>
- [4] - SACHAN, A. Tensorflow Tutorial 2: image classifier using convolutional neural network.2017. Url: https://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/ </br>
