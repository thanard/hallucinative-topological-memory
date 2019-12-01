# Hallucinative Topological Memory

Hallucinative Topological Memory (HTM) learns an energy-based model based on contrastive loss and a conditional VAE model that generates samples given a context image of a new domain. It uses these hallucinated samples for nodes, and energy-based model for the connectivity to build a planning graph. HTM allows for zero-shot generalization to domain changes. 

The environment consists of a Block Wall domain (see https://openreview.net/pdf?id=BkgF4kSFPB).

## Set-Up
1. Install standard ML libraries through pip/conda.
2. Change file to execution mode by ```chmod +x scripts/collect-data.sh```
2. Run ```./scripts/collect-data.sh``` to collect data.

## Training
1. Change to execution mode and run ```./scripts/train_vae.sh```.
2. Change to execution mode and run ```./scripts/train_actor.sh```.
3. Change to execution mode and run ```./scripts/train_cpc.sh```.

## Evaluation
1. Change mode and run ```./scripts/evaluate.sh```
