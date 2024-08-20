<h1>TransOrga-plus</h1>

Organoids have great potential to revolutionize various aspects of biomedical research and healthcare. Researchers typically use the fluorescence-based approach to analyze their dynamics, which requires specialized equipment and may interfere with their growth. Therefore, it is an open challenge to develop a general framework to analyze organoid dynamics under non-invasive and low-resource settings. In this paper, we present a knowledge-driven deep learning system named TransOrga-plus to automatically analyze organoid dynamics in a non-invasive manner.

Given a bright-field microscopic image, TransOrga-plus detects organoids through a multi-modal transformer-based segmentation module. To provide customized organoid analysis, a biological knowledge-driven branch is embedded into the segmentation module which integrates biological knowledge, e.g., the morphological characteristics of organoids, into the analysis process. Then, based on the detection results, a lightweight multi-object tracking module based on the decoupling of visual and identity features is introduced to track organoids over time. Finally, TransOrga-plus outputs the dynamics analysis to assist biologists for further research.

To train and validate our framework, we curate [a large-scale organoid dataset](https://github.com/dev-csftan/TransOrga/new/main) encompassing diverse tissue types and various microscopic imaging settings. Extensive experimental results demonstrate that our method outperforms all baselines in organoid analysis. %Moreover, we invited a group of biologists to further evaluate the effectiveness of TransOrga-plus through a real-world study. 

The results show that TransOrga-plus provides better analytical results and significantly accelerates their work process. In conclusion, TransOrga-plus enables the non-invasive analysis of various organoids from complex, low-resource, and time-lapse situations.

<h2>Requirements</h2>

<h2>How to use TransOrga-plus</h2>
<h3>Step 0: Prerequisties</h3>
<h3>Step 1: Prepare the environment</h3>
<h3>Step 2: Train the model</h3>
<h4>Step 3: Inference</h3>
