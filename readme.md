# deepseek mHC 

## details

 - regular residual
 - regular hypernetwork
 - mHC from deepseek 
 - abaltions with value residual

## performance and ablations

## possible uses

 - maybe this helps improve information to MTP heads?
 - possibly sparisfy attn farther because you can pass signals between layers cleaner and more directly now?
 - possibly improve the SWA/global mechanic, by account for SWAs weaknesses and letting the global layer dominate the hypernetwork streams?

## citations

```
@article{zhou2024valueresidual,
  title={Value Residual Learning},
  author={Zhou, Zhanchao and Wu, Tianyi and Jiang, Zhiyun and Obeid, Fares and Lan, Zhenzhong},
  journal={arXiv preprint arXiv:2410.17897},
  year={2024}
}
```

```
@software{fal_diffusion_speedrun,
  author = {{fal.ai}},
  title = {Diffusion-Speedrun: Focused fast experimentation for diffusion models},
  url = {https://github.com/fal-ai/diffusion-speedrun},
  year = {2024},
  note = {GitHub repository}
}
```

```
@article{xie2025mhc,
  title={mHC: Manifold-Constrained Hyper-Connections},
  author={Xie, Zhenda and Wei, Yixuan and Cao, Huanqi and Zhao, Chenggang and Deng, Chengqi and Li, Jiashi and Dai, Damai and Gao, Huazuo and Chang, Jiang and Zhao, Liang and Zhou, Shangyan and Xu, Zhean and Zhang, Zhengyan and Zeng, Wangding and Hu, Shengding and Wang, Yuqing and Yuan, Jingyang and Wang, Lean and Liang, Wenfeng},
  journal={arXiv preprint arXiv:2512.24880},
  year={2025}
}
```
