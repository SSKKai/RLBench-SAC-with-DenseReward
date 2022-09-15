# Training SAC agent with Dense Reward

This project shows an example of training your DRL agent in RLBench and CoppeliaSim simulation environment with hand-made dense reward.

## Installation instruction

- First you need to download the CoppeliaSim simulator. RLBench requires version **4.1** of CoppeliaSim:  
    - [Ubuntu 16.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu16_04.tar.xz)
    - [Ubuntu 18.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04.tar.xz)
    - [Ubuntu 20.04](https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz)

    Add the following to your *~/.bashrc* file:

    ```bash
    export COPPELIASIM_ROOT=PATH/TO/COPPELIASIM/INSTALL/DIR
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
    export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
    ```

    Remember to source your bashrc (`source ~/.bashrc`) or zshrc (`source ~/.zshrc`) after this.

- Now install pytorch to your environment, I recommend using conda virtual env:

    ```bash
    conda create --name your_env  
    conda activate your_env  
    conda install python=3.8.8
    pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 
    ```

- Now you can install the requirements:

    ```bash
    pip install -r requirements.txt
    pip install .
    ```

You should be good to go!

## Usage  

To use this repository, you can run the following scripts:  
- Training SAC policy:  
    ```bash
    python rlbench_rl.py
    ```
  This repo provides the hand-crafted reward functions for three RLBench tasks: *ReachTarget*, *PushButton*, *CloseMicrowave*.
  The default task is *ButtonPress*, you can change the task and training hyperparameters easily in `rlench_rl.py`.
  The reward functions are defined in the function `get_reward` in `custom_env.py`, I provided two reward equations for *ButtonPress*,
  they let the robot arm learn to push buttons vertically and horizontally respectively, you can switch them in `custom_env.py`.
  I believe these examples can help you get to know how to make your own reward functions for different tasks. 
  
  You can play with new tasks by adding the task names in `task_swtich` in `utils.py`, and edit them in `custom_env.py`.

- Teleoperating robot:  
    ```bash
    python rlbench_teleop.py
    ```
  You can teleoperate the robot by keyboard. This can help you examine and verify how the reward functions work,
  and understand the observation signals. The keyboard can be used. The key mappings are the following: 

      c -> close gripper
      v -> open gripper
      f -> gripper free
      x -> reset episode (for teleop only)
      a -> move left
      d -> move right
      s -> move backward
      w -> move forward
      q -> move down
      e -> move up
      j -> look left
      l -> look right
      i -> look up
      k -> look down
      u -> rotate left
      o -> rotate right


