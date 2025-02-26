# LLMEraser

📖 Paper: Unified Parameter-Efficient Unlearning for LLMs (ICLR 2025). Paper Link:[https://arxiv.org/pdf/2412.00383](https://arxiv.org/pdf/2412.00383).

✍️ Authors: Chenlu Ding, Jiancan Wu, Yancheng Yuan, Jinda Lu, Kai Zhang, Xiang Wang, Alex Su, and Xiangnan He

🌸 This code draws on the code of [https://github.com/ljy0ustc](https://github.com/ljy0ustc), including the implementation of LLaRA (Liao et al. 2024). Thanks for their code.


### Preparation

1. Prepare the environment: 

   ```sh
   git clone https://github.com/oceanoceanna/LLMEraser.git
   cd LLMEraser
   pip install -r requirements.txt
   ```

2. Prepare the pre-trained huggingface model of LLaMA2-7B (https://huggingface.co/meta-llama/Llama-2-7b-hf).

3. Download the data and checkpoints.

4. Prepare the data and checkpoints:

   Put the data to the dir path `data/ref/` and the checkpoints to the dir path `checkpoints/`. We provide the clean and noisy data and the corresponding checkpoints on [Movielens](https://github.com/ljy0ustc) dataset.
   
### Evaluate the clean and corrupted model 
```sh
sh test_movielens.sh
```

### Calculate the parameter changes

Using influence function to calculate the parameter changes with a single A100 GPU on Movielens dataset:

```sh
sh test_attack_movielens.sh
```

Note that: set the `llm_path` argument with your own directory path of the Llama2 model; set the correct ckpt_path.

### Hyperparameters

- $x_{lr}$: Learning rate of the optimization algorithm.

- $x_{init}$: Initial value of the parameter change.

- $x_{adjust}$: Regularization term.

- $ratio$: Attach ratio.

If you have any questions, feel free to submit an issue or contact me at dingchenlu200103@gmail.com.
