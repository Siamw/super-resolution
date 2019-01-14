# Super-resolution of Sentinel-2 images: Learning a globally applicable deep neural network (DSen2)

## Abstract
* Sentinel-2 위성은 세가지 다른 공간 해상도에서 얻은 13개의 spectral bands를 가진 multi spectral image제공
* 연구 목적 : 저해상도 ( 20m 및 60m Ground Sampling Distance ) 대역을 10m GSD로 SR시켜 완벽한 데이터 큐브를 얻는 것
* CNN을 사용하여 더 낮은 해상도인 40 to 20, 360 to 60m GSD 데이터로 트레이닝하여 end to end upsampling을 수행한다. 이런식으로 실제 Sentinel-2 image를 downsampling하여 사실상 무한한 양의 train data에 접근 가능하다.
* 다양한 data sample을 사용하며, 다시 train시킬 필요 없이, super-resolution을 진행할 수 있도록 한다. 

## Introduction
* 위성영상 : multiple spectral bands with different spatial resolution - 다중 시간적 변화 없이, 유사한 조명 및 대기 조건으로 상이한 스펙트럼 밴드가 동시에 기록된다. 
* veiwing direction : 모든 대역에서 동일
* band간 co-registration 정확
* multi spectral & multi resolution 예시들 : MODIS, VIIRS, ASTER, Worldview-3, Sentinel-2
*  
