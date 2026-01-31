# UA-CD
UA-CD: A Plug-and-play Framework for Uncertainty-aware Change Detection

> Anonymous Authors

## :round_pushpin: Todo

- [x] Release training and inference codes. Instructions on dataset preparation and checkpoints are also provided.

## :sparkles: Highlight

- **A plug-and-play uncertainty-aware CD framework.** We proposed a Change Detection framwork, which realizes the integration of uncertainty modeling into existing methods with minimal effort. 
- **uncertainty-guided regularization term.** We present an uncertainty-guided regularization term for network fine-tuning, which enhances discriminative representations and increases CD performance.
- **Extensive experiments on four challenging benchmarks.** Experiments with three representative methods on four benchmark datasets exhibit the effectiveness and universality of our proposed framework.

## :memo: Introduction

- We proposes a general plug-and-play framework for uncertainty-aware change detection, which can integrate uncertainty modeling into popular CD networks with minimal modifications. Our framework features MC-dropout to induce predictive variations and introduces a regularization term for uncertainty-aware finetuning, which brings consistent increments to baseline models. Comprehensive ablations and experiments on four benchmarks and three baseline models verify the effectiveness and universality of our framework.

  <img src="./fig/architecture.png" alt="image" style="zoom: 80%;" />


