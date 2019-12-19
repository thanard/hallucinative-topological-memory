# Hallucinative Topological Memory

Hallucinative Topological Memory (HTM) tackles the problem of <em> Visual Planning </em> in which an agent learns to plan goal-directed behavior from observations of a dynamical system obtained offline, e.g., images obtained from self-supervised robot interaction. In particular, HTM learns an energy-based model based on contrastive loss and a conditional VAE model that generates samples given a context image of a new domain. It uses these hallucinated samples for nodes, and energy-based model for the connectivity to build a planning graph. HTM allows for zero-shot generalization to domain changes. 

The environment currently consists of a Block Wall domain (see https://openreview.net/pdf?id=BkgF4kSFPB) made easy for testing zero-shot generalization.

## Set-Up
1. Install standard ML libraries through pip/conda and [Mujoco](http://www.mujoco.org/).
2. Change file to execution mode by ```chmod +x scripts/collect-data.sh```
2. Run ```./scripts/collect-data.sh``` to collect data, or download the training and test data [here](https://drive.google.com/drive/folders/1Lj9cgkWhFUU0f6X-D2bbIu-MPhQ8rSfH?usp=sharing).

## Training
1. Change to execution mode and run ```./scripts/train_vae.sh```.
2. Change to execution mode and run ```./scripts/train_actor.sh```.
3. Change to execution mode and run ```./scripts/train_cpc.sh```.

## Evaluation
1. Change mode and run ```./scripts/evaluate.sh```
