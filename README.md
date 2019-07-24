# OpenNRE-PyTorch-enhanced
A version of OpenNRE-Pytorch specifically designed for my summer REU research team's project at Lehigh University.

# Project Description
Crowdsourcing is a type of strategy to provide labels for unlabeled data where multiple non-experts are paid to label these data. However, crowdsourcing has limitations with cost and quality control; given a finite amount of resources to spend on crowdsourcers, what are the data points that best represent the dataset? 

This can also be explained as follows: all datasets contain data that is exciting and valuable and data that does not offer any new insight on the data and can be considered redundant. We want to find the best data to label such that we will not be spending our resources for crowdsourcers to label redundant data.

We approach this problem by employing a number of probabilistic models. My research group was specifically tasked to use two different machine learning models: a neural network and a conditional random field. I was in charge of the neural network architecture and running different neural networks tests, while another team member was in charge of the conditional random field and running tests on that code.

The end goal for the research project is to combine the two models we have been working on to form a conditional neural field, whose description can be found in the paper https://research.cs.washington.edu/istc/lfb/paper/nips09b.pdf. This repo is dedicated to code for designing the conditional neural field.

The template code for the neural network architecture is from the original OpenNRE-PyTorch repo at https://github.com/ShulinCao/OpenNRE-PyTorch and the CRF code is from the original OpenCRF repo at https://github.com/ZYH111/OpenCRF. However, we have greatly modified both codes to specifically fit the need for our project.