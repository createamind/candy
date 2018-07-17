# Candy
Candy: Self-driving in Carla Environment.

<img src="https://github.com/createamind/candy/blob/master/screenshots/candy.png" width="500"/>

What is candy? A model with the structure: Hierarchical Observation -- Plan&Policy -- Hierarchical Actions

We use VAE/GAN/Glow for world representation, and do RL, IL, Planning, MCTS upon it.


## VAE Demo
Real:

<div>
    <img src="https://github.com/createamind/candy/blob/master/screenshots/real1.png" width="250" style="display:inline"/>
    <img src="https://github.com/createamind/candy/blob/master/screenshots/real2.png" width="250" style="display:inline"/>
</div>

Reconstructed: (With hidden state of size 50, running for 1 hour)

<div>
    <img src="https://github.com/createamind/candy/blob/master/screenshots/reconstruct1.png" width="250" style="display:inline"/>
    <img src="https://github.com/createamind/candy/blob/master/screenshots/reconstruct2.png" width="250" style="display:inline"/>
</div>


## Running Candy
(This project is still working in progress.)
* Download Carla-0.8.2 from [here][carlarelease].
* Start CarlaUE4 engine in server mode, using commands from [here][carlagithub].
* Install Carla PythonClient using `pip install ~/carla/PythonClient`.
* Install required packages:
    pip install numpy tensorflow msgpack msgpack-numpy pyyaml tqdm gym baselines
* Start the program by running:
    CUDA_VISIBLE_DEVICES=0 python main.py -m Town01 -l

[carlagithub]: http://carla.readthedocs.io/en/latest/running_simulator_standalone/
[carlarelease]: https://github.com/carla-simulator/carla/releases


## Candy Features
* Combining imitation learning and reinforcement learning.
* Use MCTS for planning.
* Plenty of auxiliary tasks for robust hidden representation.
* Persistent training process and flexible architecture.

## Todo
* Depth as input
* Prioritized replay
* PPO
* Visualize parameter transition.
* Distributed.
* Attention for Representation explanation.
* Implement MCTS.
* Ray: Change Policy Gradient to PPO(Proximal Policy Optimization), or DDPG.
* Glow.
* Speed, Depth, Orientation as inputs.
* Stop when collision.
* Policy embedding? Curiosity, Attention, Memory ?

## Code Components
* main.py: Main file. It Deals with Carla environment.
* carla_wrapper.py: Wrap main.py, buffer information for the model.
* persistence.py: Main model file. It is for building the big graph of the model.
* modules/* : Building blocks of the model, used in persistence.py.


