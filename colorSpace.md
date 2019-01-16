# Color space

## RGB-NIR
- [dataset](https://ivrl.epfl.ch/research-2/research-downloads/supplementary_material-cvpr11-index-html/)

-[Wide-Band Color Imagery Restoration for RGB-NIR Single Sensor Images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6068539/)

현재 연구에서는 가시 광선 (RGB) 및 근적외선 (NIR) 하위 스펙트럼 (400-1100 nm)에서 작동하는 MSI 카메라에 중점을 둡니다.

근적외선 스펙트럼 밴드는 시각적으로 유사한 샘플이 매우 다른 특성을 나타낼 수있는 인간의 시각 시스템 (400-700 nm)의 범위를 벗어납니다 (예 : [5,6,7,8]).

이러한 특성은 물질, 식생, 수질 오염 등과 같은 요소를 특성화하기 위해 서로 다른 스펙트럼 대역이 사용되는 원격 탐사 응용 분야에서 크게 활용되어왔다 (예 : [9,10]).

가시 스펙트럼 이외의 컴퓨터 비전에서 사용되는 모든 다른 스펙트럼 밴드 중에서 근적외선은 가장 널리 탐구되어 왔는데, 한편으로는 재료 에너지 흡수 및 반사율과 관련된 고유 한 특성을 나타 내기 때문입니다.

반면에 실리콘 기반의 센서 (SiO2)는 1100 nm까지 NIR에 민감하므로 NIR 카메라는 다른 스펙트럼 대역에서 다른 기술과 작동하는 카메라에 비해 상대적으로 저렴합니다.

근적외선 (NIR) 밴드는 인간의 눈으로 감지 할 수있는 방사선에 가장 가까운 파장이기 때문에 근적외선 이미지는 가시적 인 이미지와 여러 속성을 공유합니다 (그림 1 참조).





* 일단 vdsr을 HSI에서 사용하도록 변경하고, 이후 RGB-NIR에 대하여 사용할 수 있는지 변경하여 확인한 뒤 n 차원으로 확장하는 것이 좋을 듯 하다.

* Sentinel-2 dataset에서는, 12개의 데이터인데, super resolution을 진행할 때 그리 간단하게 진행하지 않는다.. dataset의 해상도가 다른 문제 때문인데, 이 연구의 방법으로는 진행이 되지 않을 것이다. 
