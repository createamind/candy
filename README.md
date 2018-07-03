# Candy
Candy: Self-driving in Carla Environment.

 ![image](https://github.com/createamind/candy/blob/master/screenshots/candy.png)

What is candy? A model with the structure: Hierarchical Observation -- Plan&Policy -- Hierarchical Actions

## Running Candy
(This project is still working in progress.)
* Download Carla-0.8.2 from [here][carlarelease]
* Start CarlaUE4 engine in server mode, using command from [here][carlagithub] 
* run python main.py

[carlagithub]: http://carla.readthedocs.io/en/latest/running_simulator_standalone/
[carlarelease]: https://github.com/carla-simulator/carla/releases


## Candy Features
* C3D-VAE for observation abstraction.
* Combining imitation learning and reinforcement learning.
* Using MCTS for planning.
* Plenty of auxiliary tasks for robust hidden representation.
* Persistent training process and flexible architecture.
* Policy embedding? Curiosity, Attention, Memory ?
