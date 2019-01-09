# Super-Resolution of Multispectral Multiresolution Images from a single sensor

## Abstract

원격 감지 센서 : 다양한 스펙트럼 범위 (Sentinel-2  :   육지, 연안 해역의 높은 공간 해상도에서 광학 이미지 수집, MODIS)에서 서로 다른 공간 해상도의 **다중 스펙트럼 이미지**를 수집

연구 목적 : multi resolution sensor의 모든 spectral band를 sensor의 최고 해상도에서 추론하는 데 있다.
- 적응 형 (edge reserving) 정규 표현식을 사용하여 볼록한 목적함수를 최소화로서 이 문제를 공식화한다.
- regulariser는 고해상도 대역의 불연속성을 train하고 다른 bands(대역)로 전송하는 반면,
- data-fitting term은 bands(대역)별 individual blur와 downsampling을 설명한다.
- 또한 data가 저차원(lower-dimensional)의 부분공간에서 표현될 수 있다는 것을 알아내었고, problem의 차원을 줄이고, conditioning을 크게 개선하였다.
- simulated data를 사용한 실험에서, 실제 Sentinel-2 data에 대해 좋은 성능을 냈다.


## Introduction

- 원격 감지 (remote sensing)에서 multispectral bands에 따라 달라지는 multispectral images를 얻는 sensor가 점점 더 많아지고 있다. 이는 위성센서의 경우이며, MODIS, ASTER, VIIRS, Worldview-3, Sentinel-2가 예이다. ( 설계 고려사항, 센서 하드웨어 제한 및 대기 흡수 등의 추가 영향으로 포화된 신호 대 잡음비(SNR)를 달성하기 위해 다양한 채널에 대해 서로 다른 공간 분해능을 사용해야하기 때문 )
- 이런 해상도의 차이가 하드웨어 개선으로 사라지지는 않을것이기 대문에, coarser bands의 해상도를 광 신호의 구조를 이용하여 계산적으로 개선하는 것이 자연스럽다.
- 보편성을 잃지 않고, Sentinel-2(3가지 공간 해상도를 가지며 13개의 band를 가진다)에 초점을 맞춘다. (Table 1)
![Sentinel_band](./images/Sentinel_band.png)

- **본 연구의 목적은 저해상도 대역의 공간 해상도를 증가시켜 모든 대역이 동일한 최대 해상도를 갖게 하는 것이다.**

- 적외선 spectrum에서 많은 대역이 필요하기 때문에, 환경 및 기후 응용프로그램은 특히 이 접근법의 이점을 얻을 수 있다. 
- 이는 pan-sharpening(사진 선명화 ; 실제 해상도는 고정. Low-res color bands + High-res grayscale band = High-res color image)와의 차이점 
    1. 가장 높은 해상도에서 하나 이상의 channel이 존재할 수 있다
    2. 고해상도 bands가 저해상도 spectrum과 spectrum적으로 겹치지 않아도 된다
    - 하지만 이것만으로는 문제를 해결할 수 없다.따라서 더 다양한 spectral sensitivity와 공간 해상도의 channel을 compact한 imagin model에 합침으로서 한걸음 더 나아간다.
    
    ### Contribution : High -  Quality solution for "Smart Upsampling"
    
   -  blurring이나 downsampling문제를 감안할 때, 모델의 반전(inversion)은 ill-posed problem이다.
   - 이를 극복하기 위해 spectral bands들이 서로 상관되어있기 때문에 대부분의 에너지가 포함되어있는 낮은 차원의 부분공간(lower-dimensional subspace)에서 표현 될 수 있다는 사실을 이용한다. 
   - subspace : parameter수를 줄이고, 게산을 안정화하도록 input data로부터 학습시킨다. 
   - 또한 high-resolution band의 texture 정보를 완전히 활용하여 data의 불연속성을 인코딩하고, 공간 정보를 low resolution bands로 전파함으로써 정규 표현식을 조정해준다.
   - 내재된 주된 가정은, 불연속성이 모든 대역에서 동일한 위치에 있을 가능성이 크다는 것이다.
   - 2차 data항과 적응적인 2차 조정을 고려하여, 관측 모델을 invert시키기 위해 convex(볼록)문제를 공식화하고, 효율적인 수치적 해법을 제안한다.
   - 이 방법은, single step에서 모든 저해상도 대역에 대한 super-resolution을 최대 해상도로 수행한다.
   - "SupReME - SUPer-REsolution for Multiresolution Estimation" ; model name
   
   
   
## Related work (아직 정확한 이해는 하지 못함)

- Tonooka[16] 논문에서는, 가시광 및 근적외선 대역을 사용하여 ASTER의 열 적외선 및 단파 적외선 대역에 대해 스펙트럼 유사성을 이용하여 super-resolution을 수행한다. 
- Aizaai[2] 논문에서는, 일반화된 Laplacian Pyramid 방법을 사용하여 VNIR channel로부터 공간 세부 사항을 주입하여 ASTER TIR channel의 super resolution을 해결한다. 이는 또한 해결을 위해 일반적인 Bayesian data fusion 접근을 제안한다.
- Sirguey[15] 논문에서는, ARSIS 개념을 기반으로 한 wavelet 기반 multiresolution 분석을 injection-type 방법으로 사용하였다.
- Trishchenko[17]은 super-resolved channel의 radiometric 일관성을 보존하기 위해 비선형 회귀와 정규화 제안
- wang[20]은 regression 모델링에서 출발한 ATPRK라고 하는 fan-sharpening 방법을 제한하고, 저해상도 대역의 spectral property들을 준수하기 위해 residual upsampling을 이용합니다. 이는 하나의 고해상도 이미지만 허용하기 때문에, 스펙트럼에 가장 가가운 10m band 또는 10m band의 평균을 사용하는 것이 좋다.
- Du[6]는 S2 data의 SWIR band(B11)를 sharpening하기위한 4가지의 다른 pan-sharpening 방법을 비교한다.
- Vaiopoulos,Karantzalos[18]은 21개의 fusion 알고리즘을 비교하여 S2의 VNIR과 SWIR bands를 선명하게 했다.


## Problem Formulation
loss of generality 없이, 우리는 Sentinel-2 data를 위한 모델을 제시한다.

* Input = L = L1 + L2 + L6 spectrual bands
    * L1 = GSD 10m를 가지는 4개의 high resolution bands
    * L2 = GSD 20m를 가지는 6개의 medium resolution bands
    * L3 = GSD 30m를 가지는 3개의 low rewolution bands
    
* Output  : 모든 대역에 대해 10m 해상도를 갖도록 한다. 
    * upsampling factor : r1 = 1, r2 = 2, r6 = 6
    * fixed image area contains : n = n1 = n/(r1^2) high resolution pixels,
                                                        n2 = n/(r2^2) medium resolution pixels,
                                                        n3 = n/(r6^2) low resolution pixels

![problem_formulation](./images/problem_formulation.png)

이 논문의 방법을 도출하기 위해 **image를 벡터화**하는 것이 편리하다. 
- 각각의 개별 band의 pixel intensities : vector yi 로 수집, 이는 임의의 고정된 순서로 연결(1)된다.
- 유사하게, 알려지지 않은 output images는 (2)이지만, 모든 대역은 동일한 최대의 해상도 (3)을 갖는다.
- 따라서,  output bands는 필요하다면 matrix로 변경될 수 있다. (4)
                                                        

    
    
