# Project Brief

## Summary

This repository presents a deep learning research project for Kaggle's Vesuvius Challenge Surface Detection competition. The project focuses on 3D segmentation of scroll CT data using specialist model ensembles and topology-aware inference.

## Why It Is Technically Interesting

The hard part is not only segmenting a surface. The hard part is avoiding structural failures:

- false bridges between nearby surfaces
- broken surfaces caused by over-pruning
- unstable thresholds
- memory-heavy 3D inference
- strict runtime limits

## My Contribution

The repository documents:

- a curated model training workflow
- specialist ensemble design
- production-style inference checks
- metadata-driven experiment tracking
- reproducibility and data-governance rules
- lightweight tested utilities extracted from the notebook workflow

## Skills Demonstrated

- PyTorch-based 3D deep learning
- segmentation research
- model validation and experiment design
- performance-aware inference
- repository hygiene
- testable ML infrastructure
- technical communication

## Relevance to Research Engineering Roles

The project maps naturally to research engineering expectations in high-performance environments:

- make tradeoffs explicit
- validate assumptions
- keep systems reproducible
- separate heavyweight artifacts from source code
- design graceful degradation under resource constraints
