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
|                        |                                                                                                               | 2115  | 0.471768 | 0.823923 |
|                        |                                                                                                               | 3000  | 0.489362 | 0.835788 |
| resnet50               | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.290371 | 0.708674 |
|                        |                                                                                                               | 200   | 0.346972 | 0.750682 |
|                        |                                                                                                               | 500   | 0.391571 | 0.784097 |
|                        |                                                                                                               | 1000  | 0.425259 | 0.811102 |
|                        |                                                                                                               | 1500  | 0.439034 | 0.814512 |
|                        |                                                                                                               | 1625  | 0.458674 | 0.812057 |
|                        |                                                                                                               | 3000  | 0.484452 | 0.830742 |
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
|                        |                                                                                                               | 3000  |
| resnet50               | batch_size=32 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 50    | 0.218358 | 0.600791 |
|                        | niter=2000 <br>  niter_decay=2000                                                                             | 100   | 0.251637 | 0.657119 |
|                        |                                                                                                               | 200   | 0.290507 | 0.694490 |
|                        |                                                                                                               | 500   | 0.360474 | 0.759138 |
|                        |                                                                                                               | 1000  | 0.393617 | 0.774959 |
|                        |                                                                                                               | 1500  | 0.415030 | 0.812330 |
|                        |                                                                                                               | 2000  | 0.424168 | 0.810420 |
|                        |                                                                                                               | 2500  | 0.443944 | 0.823786 |
|                        |                                                                                                               | 3000  | 0.458811 | 0.825423 |
|                        |                                                                                                               | 3500  | 0.469722 | 0.835106 |
|                        |                                                                                                               | 3965  | 0.483088 | 0.844926 |
|                        |                                                                                                               | 3985  | 0.485543 | 0.843017 |
|                        |                                                                                                               | 4000  | 0.476678 | 0.842335 |
| resnet50 hardest       | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 100   | 0.291462 | 0.710311 |
|                        | niter=2000 <br>  niter_decay=2000                                                                             | 200   | 0.344517 | 0.749454 |
|                        |                                                                                                               | 500   | 0.396208 | 0.785052 |
|                        |                                                                                                               | 1000  | 0.437397 | 0.810829 |
|                        |                                                                                                               | 1500  | 0.449809 | 0.820649 |
|                        |                                                                                                               | 2000  | 0.447900 | 0.815057 |
|                        |                                                                                                               | 2500  | 0.466721 | 0.825968 |
|                        |                                                                                                               | 3000  | 0.471086 | 0.832788 |
|                        |                                                                                                               | 3500  | 0.480360 | 0.828833 |
|                        |                                                                                                               | 3700  | 0.493316 | 0.839198 |
|                        |                                                                                                               | 3800  | 0.488680 | 0.836607 |
|                        |                                                                                                               | 3900  | 0.491680 | 0.839607 |
|                        |                                                                                                               | 3950  | 0.493453 | 0.841653 |
|                        |                                                                                                               | 3965  | 0.493726 | 0.840016 |
| resnet50 hardest       | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_decay=0.0 | 2920  | 0.483906 | 0.834970 |
|                        | niter=2000 <br> niter_decay=2000 <br> epoch_count=2500(from resnet50 b64 best?)                               | 3175  | 0.489498 | 0.835788 |
|                        |                                                                                                               | 3195  | 0.491134 | 0.834152 |
|                        |                                                                                                               |       |
| resnet50               | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_tri=10.0  | 50    | 0.305237 | 0.683170 |
|                        | weight_decay=0.0 <br> niter=2000 <br>  niter_decay=2000                                                       | 100   | 0.361702 | 0.749182 |
|                        |                                                                                                               | 200   | 0.399891 | 0.783552 |
|                        |                                                                                                               | 500   | 0.440262 | 0.814921 |
|                        |                                                                                                               | 1000  | 0.460174 | 0.824741 |
|                        |                                                                                                               | 1500  | 0.463584 | 0.824468 |
|                        |                                                                                                               | 2000  | 0.466176 | 0.820513 |
|                        |                                                                                                               | 2500  | 0.478178 | 0.826650 |
|                        |                                                                                                               | 3000  | 0.486088 | 0.833742 |
|                        |                                                                                                               | 3895  | 0.509820 | 0.850655 |
|                        |                                                                                                               | 3500  | 0.491135 | 0.841517 |
|                        |                                                                                                               | 4000  | 0.506001 | 0.848882 |
| resnet50               | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_tri=50.0  | 50    | 0.324741 | 0.693126 |
|                        | weight_decay=0.0 <br> niter=2000 <br>  niter_decay=2000                                                       | 100   | 0.381888 | 0.764594 |
|                        |                                                                                                               | 200   | 0.425941 | 0.792690 |
|                        |                                                                                                               | 500   | 0.454719 | 0.817512 |
|                        |                                                                                                               | 1000  | 0.456901 | 0.828833 |
|                        |                                                                                                               | 1500  | 0.476678 | 0.835243 |
|                        |                                                                                                               | 2000  | 0.472450 | 0.828833 |
|                        |                                                                                                               | 2500  | 0.490180 | 0.836197 |
|                        |                                                                                                               | 3000  | 0.496727 | 0.837152 |
|                        |                                                                                                               | 3500  | 0.499864 | 0.844108 |
|                        |                                                                                                               | 3955  | 0.513775 | 0.847245 |
|                        |                                                                                                               | 3970  | 0.514048 | 0.849018 |
|                        |                                                                                                               | 3990  | 0.514594 | 0.846972 |
|                        |                                                                                                               | 4000  | 0.512275 | 0.846972 |
| resnet50               | batch_size=64 <br> lr=1e-05 <br> margin=0.3 <br> metric='euclidean' <br> weight_cat=1.0 <br> weight_tri=100.0 | 50    | 0.322013 | 0.698309 |
|                        | weight_decay=0.0 <br> niter=2000 <br>  niter_decay=2000                                                       | 100   | 0.383797 | 0.759956 |
|                        |                                                                                                               | 200   | 0.435215 | 0.796099 |
|                        |                                                                                                               | 500   | 0.462084 | 0.824468 |
|                        |                                                                                                               | 1000  | 0.470676 | 0.829242 |
|                        |                                                                                                               | 1500  | 0.478314 | 0.831697 |
|                        |                                                                                                               | 2000  | 0.477632 | 0.828287 |
|                        |                                                                                                               | 2500  | 0.488407 | 0.834288 |
|                        |                                                                                                               | 3000  | 0.500954 | 0.839470 |
|                        |                                                                                                               | 3500  | 0.499591 | 0.843562 |
|                        |                                                                                                               | 3965  | 0.513639 | 0.851064 |
|                        |                                                                                                               | 4000  | 0.511047 | 0.848472 |
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
|                        |                                                                                                               | 2500  | 0.516912 | 0.851064 |
|                        |                                                                                                               | 3420  | 0.518958 | 0.856792 |
|                        |                                                                                                               | 3500  | 0.523322 | 0.858292 |