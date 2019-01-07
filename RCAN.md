# Image Super-Resolution Using Very Deep Residual Channel Attention Networks (RCAN)

CNN의 깊이는 SR에 있어서 매우 중요하지만, 네트워크가 깊어질수록 훈련이 어려워진다. 저해상도의 입력들은 low-frequency 정보를 가지고 있고, 
이는 channel을 통과하면 동일하게 다루어지기 때문에, CNN의 표현력을 방해한다. 이 문제를 해결하기 위해, residual 구조 및 attention 네트워크를 사용한다.

residual structure를 사용함으로서, 저주피수 정보를 우회하고, 네트워크가 고주파수 정보를 학습하는 데에 중점을 둔다.
attention mechanism을 사용함으로서, 채널 간 상호 의존성을 고려하여 채널별 특성을 적용적으로 재조정 한다.

이 논문(RCAN)은 이전 연구들보다 높은 정확성과 시각적 개선을 달성하였다.

## Introduction

ill-posed problem인 SR문제를 어떻게 효과적으로 해결할 것인가? 
  1. CNN을 사용한 SRCNN
  2. Deep한 구조를 갖는 VDSR, DRCN
  3. Residual Learning 도입 ( low resolution image는 deep한 구조를 가질수록 가지는 데이터 값들이 무의미하다. high resolution 정보를 사용하자 ! )
  4. very wide한 EDSR, very deep한 MDSR -> 좋은 성능을 보임으로서, SR에 깊이가 매우 중요함을 알림.
  
  하지만, 더 깊은 네트워크를 구성하기 위해 residual block을 쌓는 것 만으로는 더 좋은 결과를 낼 수 없다.
  따라서 더 깊은 네트워크가 SR에 기여를 하는지, 어떻게 학습 가능한 deep한 구조를 train할 것인지는 탐구되어야 한다.
  또한 최근의 CNN기반 방법은 channel별 기능을 동등하게 취급하기 때문에, 다양한 유형의 정보( 저주파수 및 고주파수 정보 )를 처리할 수 있는 유연성이 부족하다.
  즉 결국 불필요한 계산을 하며, feature channel에서 차별적인 학습 능력을 결여시키고, 결국 깊은 네트워크의 표현력을 방해한다.
  
  - 이 문제들을 해결하기 위해, Residual Channel Attention Network(RCAN)를 제안한다.
    - 깊은 네트워크 훈련을 위해 residual in residual 구조(residual group이 기본 모듈로 사용되고, log skip connection이 residual learning을 사용) 제안
    - 각 residual group에서 short skip connection(SSC)을 사용하여 몇개의 단순화된 residual block을 쌓는다.
    - 채널간 상호 의존성을 모델링하는 channel attention mechanism 제안 ( 유용한 channel에 집중하고, 차별 학습 능력을 향상시킴 )
    
    
## Related Work

1. Deep CNN for SR
  - SRCNN, VDSR, DRCN, DRRN( recursive blocks ), MemNet( memory block ) : 이 방법들은 LR을 원하는 크기로 interpolation해야하고, 이는 세부사항을 잃어버리고 계산을 크게 증가시킨다.
  - FSRCNN, SRResNet, SRGAN
  - EDSR, MDSR
  
2. Attention mechanism
  - input의 요소 정보를 가장 잘 나타낼 수 있도록 하는 guidance
  - 고추파 channel별 특징은 HR 재구성에 보다 많은 정보 제공
  
## RCAN
  ### Network architecture
  ![RCAN structure](./images/RCAN.png)
  - 크게 4가지 부분으로 구성된다.
    1. shallow feature extraction : shallow feature F(0) 추출 위해 오직 하나의 convolution layer 사용
    2. deep feature extraction by residual in residual (RIR) : F(0)사용하여 진행, 결과물 F(DF)
    3. upscale module : F(DF)를 upscale하여 f(UP)를 만든다. (up scale module : deconvolution, nearest-neighbor upsampling + convolution, ESPCN 방법 존재)
    4. reconstruction part
  
  I(LR) : RCAN의 input
  I(SR) : RCAN의 output
  
