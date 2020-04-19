# Reinforcement Learning | Continuous Control | Deep Deterministic Policy Gradient (DDPG) agent | Unity Reacher (robot arm) environment
---
This notebook, shows you how to implement and train an actor-critic [DDPG](https://arxiv.org/abs/1509.02971) (Deep Deterministic Policy Gradient) Reinforcement Learning agent to steer double-jointed robot arms towards target locations in a Unity simulation environment called [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). This README.md is for the interested reader who wants to clone and run the code on her/his own machine to understand the learning algorithma and how to solve this task checkout the `Continuous_Control.ipynb` notebook.

**Why?** Reinforcement Learning (RL) is one of the most fascinating areas of Machine Learning! You might have heared about the breakthrough application [AlphaGo by DeepMind](https://deepmind.com/research/case-studies/alphago-the-story-so-far) which competed in the ancient game of [Go](https://en.wikipedia.org/wiki/Go_(game)) against the (16 times) world champion Lee Sedol and ultimately won 4-1.  RL is quite intuitive, because we use positive and negative feedback to learn tasks via interaction - just like e.g. we would train a dog. From controlling robots to stock trading to solving complex simulated tasks, RL has many applications and bears lots of potential. This notebook is a project submission for the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). My contribution lays mainly in section 1, 5, and 6.

**What?** The [Unity](https://en.wikipedia.org/wiki/Unity_(game_engine)) environment shown in this notebook is called [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md). In this episodic task the agent can control the movement of 20 double-jointed robot arms and the goal is to move each robot hand into a green target location and keep it there. If this is achieved the agent receives positive reward of +0.1 each time step and ultimately a positive game core. This simulation has 20 agents. Each agent observes a state with length 33 corresponding to position, rotation, velocity, and angular velocities of the arm:

```Python
[ 0.00000000e+00 -4.00000000e+00  0.00000000e+00  1.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -1.00000000e+01  0.00000000e+00
  1.00000000e+00 -0.00000000e+00 -0.00000000e+00 -4.37113883e-08
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  5.75471878e+00 -1.00000000e+00
  5.55726624e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
 -1.68164849e-01]
 ```

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector must be a number between -1 and 1. Have a look at a trained DDPG agent underneath.

<img src="imgs/trained_agent.gif\" width="450" align="center" title="Reacher Unity environment">

## Solving the Environment

For solving the environment the agent must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an average score for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

**How?** Checkout the [`Continuous_Control.ipynb`] and learn more about the DDPG algorithm, the implementation and how to train an agent.


## Dependencies

Set up your python environment to run the code in this repository, follow the instructions below.

1. Create and activate a new conda environment with Python 3.6. If you don't have *Conda*, click here for [Conda installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/). 

	- __Linux__ or __Mac__: 
	```bash
	conda create --name ddpg python=3.6
	source activate ddpg
	```
	- __Windows__: 
	```bash
	conda create --name ddpg python=3.6 
	activate ddpg
	```

2. Clone this repository, and navigate to the `ddpg-continuous-control/python/` folder.  Then, install several dependencies related to the Unity environment. Check the dir for details.
```bash
git clone https://github.com/ImmanuelXIV/ddpg-continuous-control.git
cd ddpg-continuous-control/python
pip install .
```

3. Download the Reacher environment from one of the links below. You only need to select the environment that matches your operating system. Place it in the `ddpg-continuous-control/` dir, decompress it and change the `file_name` in the `Continuous_Control.ipynb` section 2 accordingly. 

Twenty (20) Agents Reacher Unity environment

Downloads
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Paths
- **Mac**: `"path/to/Reacher.app"`
- **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
- **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
- **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
- **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
- **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
- **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

For instance, if you are using a Mac, then you downloaded `Reacher.app`.  If this file is in the same folder as the notebook, then the line in section 2 in the notebook should appear as follows:
```
env = UnityEnvironment(file_name="Reacher.app")
```

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use this link [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)


4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `ddpg` environment.  
```bash
python -m ipykernel install --user --name ddpg --display-name "ddpg"
```

5. Run the following code and follow the instructions in the notebook (e.g. run all)
```bash
cd ddpg-continuous-control/
jupyter notebook
```

6. Before running code in the `Continuous_Control.ipynb` notebook, change the kernel to match the `ddpg` environment by using the drop-down `Kernel` menu in the toolbar. 