# Awesome Dexterous Manipulation Papers

A comprehensive collection of research papers on dexterous manipulation, focusing on algorithmic advances while including key hardware, sensing, and systems contributions. Papers span 2020-2025 from top robotics and ML conferences, plus seminal classical works.

## Table of Contents

### Learning and Control Methods
1. [Reinforcement Learning Methods](#reinforcement-learning-methods)
2. [Imitation Learning Approaches](#imitation-learning-approaches)
3. [Model-Based Control](#model-based-control)
4. [Vision-Language-Action (VLA) Models](#vision-language-action-vla-models)

### Sensing and Perception
5. [Vision-Based Manipulation](#vision-based-manipulation)
6. [Tactile Sensing Integration](#tactile-sensing-integration)

### Manipulation Algorithms and Tasks
7. [Grasp Synthesis Algorithms](#grasp-synthesis-algorithms)
8. [Contact-Rich Manipulation](#contact-rich-manipulation)
9. [Bimanual Coordination](#bimanual-coordination)

### Deployment and Transfer
10. [Sim-to-Real Transfer](#sim-to-real-transfer)

### Historical Perspectives
11. [Recent Breakthrough Papers](#recent-breakthrough-papers)
12. [Classical/Foundational Papers](#classicalfoundational-papers)

---

## Reinforcement Learning Methods

### Deep RL for Dexterous Control

**Learning Dexterous In-Hand Manipulation**
- Authors: OpenAI (Marcin Andrychowicz, Bowen Baker, Maciek Chociej, et al.)
- Venue: International Journal of Robotics Research, 2020
- First successful sim-to-real transfer of vision-based dexterous in-hand manipulation using Shadow Hand. Used extensive domain randomization with PPO to achieve complex behaviors without human demonstrations.

**MyoDex: A Generalizable Prior for Dexterous Manipulation**
- Authors: Vittorio Caggiano, Sudeep Dasari, Vikash Kumar
- Venue: ICML 2023
- Developed generalizable behavioral priors using physiologically realistic musculoskeletal hand model. Multi-task learning captures task-agnostic priors enabling few-shot generalization and 4x faster learning.

**Bi-DexHands: Towards Human-Level Bimanual Dexterous Manipulation with Reinforcement Learning**
- Authors: Yuanpei Chen, Tianhao Wu, Shengjie Wang, et al. (Peking University)
- Venue: NeurIPS 2022 (Datasets and Benchmarks Track)
- Comprehensive benchmark for bimanual manipulation with 30,000+ FPS training in Isaac Gym. Demonstrated PPO can solve tasks equivalent to 48-month human development stages.

**DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation**
- Authors: Yuzhe Qin, Binghao Huang, Zhao-Heng Yin, Hao Su, Xiaolong Wang
- Venue: CoRL 2022
- First category-level generalization for dexterous manipulation using point clouds. Novel contact-based rewards enable joint learning across multiple objects with successful sim-to-real transfer.

### Multi-Task and Transfer Learning

**DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training**
- Authors: Aleksei Petrenko, Arthur Allshire, Gavriel State, Ankur Handa, Viktor Makoviychuk
- Venue: RSS 2023
- Decentralized Population-Based Training for learning complex manipulation skills including regrasping, grasp-and-throw, and object reorientation using parallel GPU simulation.

**Deep Dynamics Models for Learning Dexterous Manipulation**
- Authors: Anusha Nagabandi, Kurt Konolige, Sergey Levine, Vikash Kumar
- Venue: CoRL 2020
- Combined model-based RL with learned dynamics models. Hierarchical reward structures separate low-level contact dynamics from high-level task objectives for improved learning efficiency.

---

## Imitation Learning Approaches

### Behavioral Cloning and Diffusion Policies

**Diffusion Policy: Visuomotor Policy Learning via Action Diffusion**
- Authors: Cheng Chi, Zhenjia Xu, Siyuan Feng, et al.
- Venue: RSS 2023, IJRR 2024
- Introduces diffusion models for action generation in manipulation, treating actions as noisy samples to be denoised. Enables multimodal action distributions and robust behavior learning.

**Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ALOHA/ACT)**
- Authors: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn
- Venue: RSS 2023
- Action Chunking with Transformers (ACT) algorithm for bimanual dexterous manipulation. Achieves 80-90% success on difficult tasks with only 10 minutes of demonstrations.

**3D Diffusion Policy (DP3): Generalizable Visuomotor Policy Learning**
- Authors: Yanjie Ze, Gu Zhang, Kangning Zhang, et al.
- Venue: RSS 2024
- Incorporates 3D representations into diffusion policies achieving 24.2% relative improvement. Shows excellent generalization across space, viewpoint, and appearance with 85% real robot success rate.

### Teleoperation and Human Demonstrations

**DexCap: Scalable and Portable Mocap Data Collection System for Dexterous Manipulation**
- Authors: Chen Wang, Haochen Shi, Weizhuo Wang, et al.
- Venue: RSS 2024
- Portable motion capture combining SLAM and electromagnetic tracking for precise hand motion. DexIL algorithm achieves 3x faster data collection than traditional teleoperation.

**AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System**
- Authors: Yuzhe Qin, Wei Yang, Binghao Huang, et al.
- Venue: RSS 2023
- General teleoperation framework for various dexterous robots using vision-based tracking and retargeting, enabling efficient demonstration collection across different embodiments.

**DexMV: Imitation Learning for Dexterous Manipulation from Human Videos**
- Authors: Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, et al.
- Venue: ECCV 2022
- Pipeline for learning from human videos through 3D hand-object pose estimation and motion retargeting, bridging the gap between human demonstrations and robot execution.

---

## Model-Based Control

### Model Predictive Control (MPC)

**Contact-Implicit Model Predictive Control for Dexterous In-hand Manipulation: A Long-Horizon and Robust Approach**
- Authors: Yongpeng Jiang, Mingrui Yu, Xinghao Zhu, et al.
- Venue: IROS 2024
- Novel contact-implicit MPC running at 20Hz on 23-DOF tasks. Enables robust long-horizon manipulation without predefined contact sequences through replanning.

**Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity**
- Authors: Sharanya Venkatesh, Bibit Bianchini, Michael Posa, et al.
- Venue: ArXiv 2025
- Globally-informed controller combining local complementarity control with global sampling. Addresses limitations of purely local methods for real-time dexterous manipulation.

**Vision-Language Model Predictive Control for Manipulation Planning**
- Authors: Jiaming Chen, Yucheng Hu, Zhenggang Tang, et al.
- Venue: ArXiv 2025
- Integrates vision-language models with MPC for environmental perception. Uses conditional action sampling and video prediction for future state simulation.

### Action Primitives and Hierarchical Control

**Accelerating Robotic Reinforcement Learning via Parameterized Action Primitives (RAPS)**
- Authors: Murtaza Dalal et al.
- Venue: NeurIPS 2021
- Demonstrates how parameterized action primitives accelerate learning by reducing action space complexity. Agents learn to select primitive categories and parameters, showing substantial improvement across 250+ evaluated agents.

**APriCoT: Action Primitives based on Contact-state Transition for In-Hand Tool Manipulation**
- Authors: Shota Saito, Tatsuya Kanehira
- Venue: ArXiv 2024
- Decomposes tool manipulation into discrete contact states with parameterized transitions. Achieves superior performance by explicitly modeling relationships between high-level contact goals and low-level joint trajectories.

---

## Vision-Language-Action (VLA) Models

### Hierarchical VLA Approaches

**DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping**
- Authors: Yifan Zhong et al.
- Venue: ArXiv 2025
- Implements hierarchical (category, parameter) action space using Vision-Language models for grasp category selection with diffusion-based parameter generation. Achieves 90% success in zero-shot cluttered environments.

**Vision-Language-Action Model and Diffusion Policy Switching Enables Dexterous Control of an Anthropomorphic Hand**
- Authors: Cheng Pan et al.
- Venue: ArXiv 2024
- Hybrid approach on ADAPT Hand 2 (13-DOF) using VLA for high-level grasp categories and diffusion for low-level control. Achieves 80% success vs 40% with end-to-end VLA.

**DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control**
- Authors: Various
- Venue: ArXiv 2025
- Billion-parameter diffusion-based action expert with embodiment curriculum learning. Shows 27% improvement with 85% success on novel tasks through three-stage progressive training.

### Language-Driven Grasping

**Language-driven Grasp Detection**
- Authors: An Dinh Vuong, Minh Nhat Vu, Baoru Huang, Nghia Nguyen, Hieu Le, Thieu Vo, Anh Nguyen
- Venue: CVPR 2024
- Enables natural language specification of desired grasps. Integrates language understanding with grasp detection for flexible human-robot interaction.

---

## Tactile Sensing Integration

**DIFFTACTILE: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation**
- Authors: Zilin Si, Gu Zhang, Qingwei Ben, et al.
- Venue: ICLR 2024
- First physics-based fully differentiable tactile simulation using FEM-based soft body modeling. Enables gradient-based optimization for sim-to-real transfer of tactile-assisted skills.

**3D-ViTac: Learning Fine-Grained Manipulation with Visuo-Tactile Sensing**
- Authors: Binghao Huang, Yuanpei Chen, Tianyu Wang, et al.
- Venue: CoRL 2024
- Multi-modal system featuring dense tactile sensors with 3mm² units. Demonstrates significant performance enhancement in contact-rich bimanual manipulation scenarios.

**Adaptive Visuo-Tactile Fusion with Predictive Force Attention for Dexterous Manipulation**
- Authors: Jinzhou Li, Xiaofeng Guo, Tongwei Lu, et al.
- Venue: ArXiv 2025
- Force-guided attention fusion adaptively adjusting visual and tactile feature weights. Achieves 93% success rate without human labeling of modality importance.

---

## Grasp Synthesis Algorithms

### Grasp Taxonomies and Foundations

**The GRASP Taxonomy of Human Grasp Types**
- Authors: Thomas Feix, Javier Romero, Heinz-Bodo Schmiedmayer, Aaron M. Dollar, Danica Kragic
- Venue: IEEE Transactions on Human-Machine Systems, 2016
- Comprehensive synthesis of 33 distinct grasp types organized into 17 general categories. Unified framework using opposition type, virtual finger assignments, and power/precision classification becoming de facto standard.

### Large-Scale Grasp Datasets

**DexGraspNet: A Large-Scale Robotic Dexterous Grasp Dataset**
- Authors: Ruicheng Wang, Jialiang Zhang, Jiayi Chen, et al.
- Venue: ICRA 2023
- Large-scale dataset with 1.32 million grasps for 5,355 objects using ShadowHand. Introduces efficient synthesis with accelerated differentiable force closure estimation.

**DexGraspNet 2.0: Large-Scale Synthetic Benchmark for Dexterous Grasping in Cluttered Scenes**
- Authors: Galbot Team
- Venue: CoRL 2024 Showcase
- Features 8,270 scenes and 427 million grasp labels for LEAP hand. Achieves 90.70% real-world success with zero-shot sim-to-real transfer.

**AnyDexGrasp: General Dexterous Grasping for Different Hands with Human-level Learning Efficiency**
- Authors: Hao-Shu Fang et al.
- Venue: ArXiv 2025
- Efficient approach requiring only hundreds of attempts on 40 objects. Two-stage approach with universal contact-centric model and hand-specific decision model achieving 75-95% success.

**Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy**
- Authors: Jiayi Chen et al.
- Venue: ArXiv 2025
- Efficient pipeline for contact-rich, penetration-free grasps covering 31 types in GRASP taxonomy. Dataset with 10.7k objects and 9.5M grasps achieves 82.3% real-world success.

---

## Bimanual Coordination

**BiDexHD: Learning Diverse Bimanual Dexterous Manipulation Skills from Human Demonstrations**
- Authors: Bohan Zhou, Zongqing Lu, et al.
- Venue: ArXiv 2024 (submitted to ICLR 2025)
- Unified framework with teacher-student policy learning. Achieves 74.59% success on 141 tasks with 51.07% generalization to unseen tasks from TACO dataset.

**BimanGrasp: Bimanual Grasp Synthesis for Dexterous Robot Hands**
- Authors: Yanming Shao, Chenxi Xiao
- Venue: IEEE Robotics and Automation Letters, 2024
- First large-scale bimanual grasp dataset with 150k+ verified grasps. BimanGrasp-DDPM diffusion model achieves 69.87% synthesis success with computational acceleration.

**Dynamic Handover: Throw and Catch with Bimanual Hands**
- Authors: Binghao Huang, Yuanpei Chen, Tianyu Wang, et al.
- Venue: CoRL 2023
- Addresses dynamic handover requiring precise spatial-temporal coordination. Demonstrates successful throw-and-catch behaviors with learning-based approaches.

---

## Sim-to-Real Transfer

**Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids**
- Authors: Toru Lin, Kartik Sachdev, Linxi Fan, et al.
- Venue: ArXiv 2025 (submitted to ICRA 2025)
- Novel techniques including automated real-to-sim tuning, generalized rewards, divide-and-conquer distillation. Demonstrates robust generalization without human demonstrations.

**DextrAH-RGB: Visuomotor Policies to Grasp Anything with Dexterous Hands**
- Authors: Ritvik Singh, Jing Xu, Yijin Yang, et al.
- Venue: ArXiv 2024
- First end-to-end RGB-based policy for complex dexterous grasping with robust sim-to-real transfer. Uses fabric-guided policy with photorealistic tiled rendering distillation.

**Transferring Dexterous Manipulation from GPU Simulation to a Remote Real-World TriFinger**
- Authors: Arthur Allshire, Mayank Mittal, Varun Lodaya, et al.
- Venue: IROS 2022
- Successful sim-to-real transfer using Isaac Gym and extensive domain randomization. Robust policies for object reorientation on real TriFinger robot.

---

## Vision-Based Manipulation

**Learning Visuotactile Estimation and Control for Non-prehensile Manipulation under Occlusions**
- Authors: Juan Del Aguila Ferrandis, Joao Moura, Sethu Vijayakumar
- Venue: CoRL 2024
- Addresses object occlusions in contact-rich manipulation using visuotactile state estimators and uncertainty-aware control policies.

**PoCo: Policy Composition from and for Heterogeneous Robot Learning**
- Authors: Lirui Wang, Jialiang Zhao, Yilun Du, et al.
- Venue: RSS 2024
- Flexible approach combining information across modalities and domains using diffusion models. Enables task-level composition for multi-task manipulation.

---

## Contact-Rich Manipulation

**Diff-LfD: Contact-aware Model-based Learning from Visual Demonstration**
- Authors: Xinghao Zhu, JingHan Ke, Zhixuan Xu, et al.
- Venue: CoRL 2023
- Combines differentiable rendering for pose/shape estimation with differentiable simulation for contact sequence generation from human video demonstrations.

**NeuralFeels: Multimodal In-Hand Perception via Neural Fields**
- Authors: Various
- Venue: Major robotics conference 2024
- Combines vision, touch, and proprioception for in-hand object pose and shape estimation. Achieves 81% reconstruction F-scores with 2.3mm average pose drift.

---

## Recent Breakthrough Papers

**Eureka: Human-Level Reward Design via Coding Large Language Models**
- Authors: Yecheng Jason Ma, William Liang, Guanzhi Wang, et al.
- Venue: ArXiv 2023, ICLR 2024
- Revolutionary use of GPT-4 for automatic reward function generation. Outperformed human experts on 83% of tasks with 52% average improvement, achieving first simulated pen spinning with Shadow Hand.

**RoboPianist: Dexterous Piano Playing with Deep Reinforcement Learning**
- Authors: Kevin Zakka, Philipp Wu, Laura Smith, et al.
- Venue: CoRL 2023
- Breakthrough enabling simulated hands to learn 150 piano pieces. Introduced comprehensive benchmark testing limits of human-level dexterity with interpretable metrics.

**HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop RL**
- Authors: Jianlan Luo et al.
- Venue: ArXiv 2024
- Vision-based RL achieving near-perfect success on diverse tasks within 1-2.5 hours. First to achieve dual-arm coordination with image inputs using RL in real-world settings.

**DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation**
- Authors: Zhenyu Jiang, Yuqi Xie, Kevin Lin, et al.
- Venue: ArXiv 2024, ICRA 2025
- Synthesizes 21K trajectories from only 60 human demonstrations. Includes real-to-sim-to-real pipeline deployed on humanoid can sorting.

**EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video**
- Authors: Ryan Hoque et al.
- Venue: ArXiv 2025
- Largest dexterous manipulation dataset with 829 hours of egocentric video using Apple Vision Pro. Covers 194 tabletop tasks with paired 3D hand tracking.

---

## Classical/Foundational Papers

### Early Dexterous Hand Designs

**Stanford/JPL Hand (1982-1987)**
- Authors: J.K. Salisbury and J.J. Craig
- Venue: International Journal of Robotics Research, 1982
- First truly dexterous robotic hand with three fingers and 9 DOF. Pioneered force feedback control establishing foundation for subsequent dexterous hand research.

**Utah/MIT Dextrous Hand (1984-1986)**
- Authors: S.C. Jacobsen, E.K. Iversen, D.F. Knutti, et al.
- Venue: IEEE ICRA 1986
- Most human-like robotic hand of its era with four fingers and 16 DOF. Established anthropomorphic design principles influencing decades of hand designs.

**DLR Hand Series (1990s-2000s)**
- Authors: J. Butterfass, M. Grebenstein, H. Liu, G. Hirzinger
- Venue: IEEE ICRA 2001
- Advanced modular design with integrated actuation and sensing. Demonstrated industrial-grade reliability bridging research prototypes and practical applications.

### Foundational Theory and Control

**Robot Hands and the Mechanics of Manipulation**
- Authors: M.T. Mason and J.K. Salisbury
- Venue: MIT Press, 1985
- Established mathematical framework for dexterous manipulation including contact mechanics, grasp statics, and force analysis. Foundation for all subsequent manipulation research.

**On the Closure Properties of Robotic Grasping**
- Authors: A. Bicchi
- Venue: International Journal of Robotics Research, 1995
- Rigorous mathematical foundations for form-closure and force-closure analysis. Standard reference for grasp analysis algorithms.

**A Mathematical Introduction to Robotic Manipulation**
- Authors: R.M. Murray, Z. Li, and S.S. Sastry
- Venue: CRC Press, 1994
- Comprehensive mathematical framework including screw theory, contact mechanics, and grasp analysis. Unified theoretical foundations becoming standard reference.

### Early Learning Approaches

**Learning from Observation Using Primitive-Based Representations**
- Authors: Y. Kuniyoshi, M. Inaba, and H. Inoue
- Venue: IEEE Transactions on Robotics and Automation, 1996
- Introduced imitation learning for manipulation, establishing foundation for learning from demonstration approaches central to modern robotics.

**Hands for Dextrous Manipulation and Robust Grasping: A Difficult Road Towards Simplicity**
- Authors: A. Bicchi
- Venue: IEEE Transactions on Robotics and Automation, 2000
- Influential survey analyzing complexity-performance trade-offs. Argued for simpler designs influencing development of underactuated hands.

---

## Resources and Benchmarks

- **Isaac Gym**: GPU-accelerated physics simulation enabling 30,000+ FPS training
- **TACO Dataset**: 141 bimanual manipulation tasks for evaluation
- **DexGraspNet**: Million+ scale grasp datasets for multiple hands
- **Real Robot Challenge**: Remote real robot experimentation platform
- **Open X-Embodiment**: Cross-embodiment dataset from 22 robots with 527 skills

## Key Trends (2020-2025)

1. **LLM Integration**: Using language models for reward design and task planning
2. **Diffusion Policies**: Robust policy learning with 3D representations
3. **Human-in-the-Loop**: Efficient real-world learning with human guidance
4. **Large-Scale Data**: Automated synthesis from minimal demonstrations
5. **Cross-Embodiment**: Generalizing across different hand morphologies
6. **Multimodal Sensing**: Vision-touch integration for robust manipulation
7. **Sim-to-Real**: Improved transfer with domain randomization and adaptation
