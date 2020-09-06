## Contador de pessoas (Definir nome)

### Equipe

| [<img src="https://avatars0.githubusercontent.com/u/31224982?s=460&u=a5c288a099540e11babe8b352e5d62ab28bae601&v=4" width="150px;"/>](https://github.com/BrunoRech) | [<img src="https://avatars2.githubusercontent.com/u/26040800?v=3&s=150" width="150px;"/>](https://github.com/MarioFronza) |
| :-------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
|                                     [Bruno Rech](https://github.com/BrunoRech)                                      |                                      [Mário Fronza](https://github.com/MarioFronza)                                       |


## Problema

O problema que a aplicação irá solucionar é o monitoramento manual da quantidade de pessoa s em um determinado estabelecimento, visto que muitos lugares necessitam controlar o número de pessoas em decorrência da pandemia do COVID-19. No cenário ideal, todo usuário deve estar utilizando máscara, sendo muitas vezes um controle manual por parte do estabelecimento.

A ideia seria utilizar um aplicativo na entrada do local. O estabelecimento deve informar ao aplicativo a quantidade máxima de pessoas permitidas. Após isso, o aplicativo detecta a face dos usuários, e verifica se o mesmo está utilizando máscara. Caso esteja, o aplicativo contabiliza o usuário, caso contrário não. Além disso, outro dispositivo deverá estar na saída do estabelecimento, contabilizando os usuários que já saíram.

O projeto futuramente poderia realizar integrações com equipamentos eletrônicos, como portas que só abririam caso o usuário estivesse utilizando máscara.

## Dataset

Será utilizado um dataset público de imagens com faces de pessoas para treinar os algoritmos. O dataset contém 1916 fotos de pessoas com máscaras e 1919 de faces sem máscaras.

[https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG](https://drive.google.com/drive/folders/1XDte2DL2Mf_hw4NsmGst7QtYoU7sMBVG)

ou este outro, com:

[https://www.kaggle.com/andrewmvd/face-mask-detection?select=images](https://www.kaggle.com/andrewmvd/face-mask-detection?select=images)

## Técnica

O problema será resolvido utilizando técnicas de redes neurais artificiais para aprender a classificar faces com máscaras. Além disso, o projeto irá utilizar a biblioteca OpenCV para encontrar faces humanas em imagens, utilizando o algoritmo Viola-Jones e sua técnica de aprendizado de máquina AdaBoost.
<!--
```shell
tflite_convert --keras_model_file=mask_detector  --output_file=mask_detector.tflite
``` -->
