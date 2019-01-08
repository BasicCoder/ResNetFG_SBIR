# Caffe2Torch
This project is to expand [CDOTAD/SketchyDatabase](https://github.com/CDOTAD/SketchyDatabase) project.
## Results
| Evaluation Model       | args                                                                                                          | epoch | Recall@1 | Recall@5 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- | ----- | -------- | -------- |
| vgg16(random split)    | batch_size=16 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 190   | 0.329451 | 0.698081 |
|                        | niter=1000                                                                                                    | 370   | 0.350364 | 0.720847 |
|                        |                                                                                                               | 480   | 0.353673 | 0.707743 |
|                        |                                                                                                               | 495   | 0.362277 | 0.719788 |
|                        |                                                                                                               | 865   | 0.370086 | 0.729054 |
|                        |                                                                                                               | 875   | 0.371674 | 0.733289 |
| resnet50(random split) | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 465   | 0.410457 | 0.802647 |
|                        |                                                                                                               | 775   | 0.432561 | 0.826737 |
|                        |                                                                                                               | 825   | 0.443150 | 0.821840 |
| resnet50(random split) | batch_size=48 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.254665 | 0.662872 |
|                        |                                                                                                               | 200   | 0.296228 | 0.708273 |
|                        |                                                                                                               | 500   | 0.344540 | 0.746923 |
|                        |                                                                                                               | 750   | 0.372733 | 0.773130 |
|                        |                                                                                                               | 1070  | 0.392455 | 0.793382 |
| resnet50               | batch_size=128<br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.281233 | 0.684806 |
|                        |                                                                                                               | 200   | 0.334288 | 0.741408 |
|                        |                                                                                                               | 500   | 0.398663 | 0.780687 |
|                        |                                                                                                               | 1000  | 0.433170 | 0.805646 |
|                        |                                                                                                               | 1500  | 0.457447 | 0.816967 |
|                        |                                                                                                               | 1770  | 0.465767 | 0.826650 |
|                        |                                                                                                               |       |          |          |
| resnet50               | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.290371 | 0.708674 |
|                        |                                                                                                               | 200   | 0.346972 | 0.750682 |
|                        |                                                                                                               | 500   | 0.391571 | 0.784097 |
|                        |                                                                                                               | 1000  | 0.425259 | 0.811102 |
|                        |                                                                                                               | 1500  | 0.439034 | 0.814512 |
|                        |                                                                                                               | 1625  | 0.458674 | 0.812057 |
|                        |                                                                                                               |       |          |          |
| resnet50               | batch_size=48 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.276732 | 0.683170 |
|                        | niter=1000 <br>  niter_decay=2000                                                                             | 200   | 0.328287 | 0.737179 |
|                        |                                                                                                               | 500   | 0.380660 | 0.776187 |
|                        |                                                                                                               | 965   | 0.433442 | 0.805237 |
|                        |                                                                                                               | 1500  | 0.440398 | 0.811784 |
|                        |                                                                                                               | 1915  | 0.462084 | 0.828287 |
|                        |                                                                                                               | 2500  | 0.472040 | 0.831560 |
|                        |                                                                                                               | 2750  | 0.484861 | 0.831015 |
|                        |                                                                                                               | 2790  | 0.488543 | 0.837289 |
|                        |                                                                                                               | 2970  | 0.487861 | 0.838925 |
| resnet152              | batch_size=48 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.254910 | 0.662029 |
|                        |                                                                                                               | 200   | 0.314375 | 0.733497 |
|                        |                                                                                                               |       |
| resnet101              | batch_size=48 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.292280 | 0.699673 |
|                        |                                                                                                               | 200   | 0.364430 | 0.769913 |
|                        |                                                                                                               | 500   | 0.424168 | 0.810829 |
|                        |                                                                                                               | 1000  | 0.446400 | 0.824332 |
|                        |                                                                                                               | 1390  | 0.480087 | 0.837561 |
|                        |                                                                                                               | 1500  | 0.464130 | 0.824195 |
|                        |                                                                                                               | 1725  | 0.486361 | 0.851746 |