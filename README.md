### This repository provides the source code of HetComp Framework (Heterogeneous model compression for recommender system)

#### Overview
Recent recommender systems have shown remarkable performance by using a large and sophisticated model. Further, an ensemble of heterogeneous models is known to achieve significantly increased accuracy over a single model. However, it is exceedingly costly because it requires resources and inference latency proportional to the number of models, which remains the bottleneck for production. Our work aims to transfer the knowledge of heterogeneous teacher ensemble to a lightweight student model using knowledge distillation (KD), so as to reduce the huge inference costs while retaining high accuracy. From our analysis, we observe that distillation from heterogeneous teachers is particularly challenging and leads to a huge discrepancy to the teachers. Nevertheless, we also show that an important signal which can help to ease the difficulty can be obtained from the teacher training trajectory. This paper proposes a new KD framework, named HetComp, that guides the student model by transferring easy-to-hard sequences of knowledge generated from the teachers' trajectory. To provide guidance according to the student's learning state, HetComp uses dynamic knowledge construction to provide progressively difficult ranking knowledge and adaptive knowledge transfer to gradually transfer finer-grained ranking information.
Our comprehensive experiments show that HetComp significantly improves the distillation quality and the generalization of the student model.

### Requirements
This is an implementation of HetComp on MF-Student.
- A-music dataset can be downloaded from: http://jmcauley.ucsd.edu/data/amazon/
- CiteULike dataset can be downloaded from: https://github.com/js05212/citeulike-t/blob/master/users.dat
- Foursquare dataset can be downloaded from: https://github.com/allenjack/SAE-NAD
- Required version: torch >= 1.10.1
