# Applying cPoI on Multi-Task Multi-Fidelity Optimization Problem (Dissertation)

This repository contains my dissertation project focused on **Applying Correlated Probability of Improvement (cPoI) on Multi-Task Multi-Fidelity Optimization Problems**. 

## Project Overview

Optimization plays a critical role in engineering and science, enabling designers and engineers to make decisions that balance multiple conflicting objectives. Multi-Objective Bayesian Optimization (MOBO) has become a powerful framework for tackling these challenges, as it efficiently identifies trade-offs without requiring exhaustive exploration of the design space.

Incorporating **multi-fidelity optimization methods**—which utilize multiple levels of approximation to evaluate objective functions—further enhances MOBO's efficiency. These methods blend **low- and high-fidelity data**, optimizing accuracy while minimizing computational cost.

### Key Focus

A central aspect of MOBO is the **maximization of the acquisition function**, which guides the search for optimal solutions. Traditional acquisition functions typically assume independence among objectives, ignoring their correlations. This project explores the application of **cPoI (Correlated Probability of Improvement)**, which explicitly accounts for correlations between objectives. The study evaluates cPoI's performance on a simulated real-world multi-fidelity optimization problem: **vehicle crashworthiness**. 

### Highlights

- Achieved a **49% improvement** in effectively capturing objective correlations, as measured by performance comparison with standard acquisition functions that do not consider correlations.
- Demonstrated the potential of cPoI to enhance **optimization efficiency and accuracy** in a multi-task, multi-fidelity context.

---

## Project Pipeline

1. **Problem Definition**: Define objectives and constraints for the optimization problem.  
2. **Initial Sampling**: Generate initial samples for model training using a multi-fidelity approach.  
3. **Surrogate Model Building**: Train Gaussian Process Regression models for each objective.  
4. **Iterative Optimization**: Maximize the cPoI acquisition function to refine solutions iteratively.  
5. **Post-Optimization Analysis**: Evaluate the performance of cPoI against baseline acquisition functions.  

---

## Tools and Techniques

- **Programming**: Python  
- **Optimization Methods**: Bayesian Optimization, Correlated Probability of Improvement  
- **Modeling**: Gaussian Process Regression, Monte Carlo Simulations  

---

This project showcases how advanced acquisition functions like cPoI can significantly improve optimization outcomes by accounting for objective correlations, paving the way for more robust and cost-effective engineering solutions.
