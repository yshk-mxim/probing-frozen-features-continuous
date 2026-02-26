# Do Foundation Models Know Geometry?

**Probing Frozen Features for Continuous Physical Measurement**

Yakov Pyotr Shkolnikov

---

## Abstract

Vision-language models encode continuous geometry that their text pathway fails to express: a 6,000-parameter linear probe extracts hand joint angles at 6.1° MAE from frozen features, while the best text output achieves only 20.0°---a 3.3x bottleneck. LoRA fine-tuning (r=16, 2,000 images) narrows this gap to 6.5°, providing evidence for a pathway-training deficit rather than a representational one.

Training objective determines accuracy more than architecture: five encoders spanning self-supervised, contrastive, and hybrid paradigms converge to statistically equivalent accuracy (R²≈0.55, TOST-equivalent at Δ=0.03) despite sharing as little as CKA=0.41 representational similarity---functional convergence without representational convergence, extending the platonic representation hypothesis to continuous geometric targets. Results validated across fourteen backbones on head pose, rigid objects, gaze, and camera intrinsics; rankings hold under nested 10-fold CV (Friedman χ²=94.3, p<10⁻¹⁵).

## Paper

The compiled paper is available in [Releases](https://github.com/yshk-mxim/probing-frozen-features-continuous/releases).

## Cite this work

```bibtex
@article{shkolnikov2026geometry,
  title={Do Foundation Models Know Geometry? Probing Frozen Features for Continuous Physical Measurement},
  author={Shkolnikov, Yakov Pyotr},
  year={2026}
}
```

## License

This work is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
