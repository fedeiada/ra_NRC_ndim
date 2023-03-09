# ra_NRC_ndim
Single thread code for ra-NRC that can handle multi dimensional cost functions,

## how to star
- run simulation.py to make an esperiment
- user can change specifications by the class SimulationSpecification in the SimulationSpecification.py file

## Documentation
You can find a presentation of the algorithms and how it works [here](https://github.com/fedeiada/ra_NRC_ndim/files/10932072/NRC-presentation.pdf). 
The algorithm implements a ratio consensus that reconstruct the global value of the gradient and hessian. The step-size is fixed. Next figure shows the evolution of the consensus signals that correctly reach an common value
![ratio_consensus_signal](https://user-images.githubusercontent.com/98212546/224038972-11d21211-5277-48fc-b49d-e5a127db7e51.png)


## Results obtained
The code was tested with randomly generated convex functions such as quadratic and exponential.
Results are the following ones:
### quadratic cost function
![quad_log](https://user-images.githubusercontent.com/98212546/224038393-71ec0c4f-b601-4792-aa3f-19619529da7f.png)
![quad_evol](https://user-images.githubusercontent.com/98212546/224038400-474117c5-16fd-4307-86d7-1331a64b30d6.png)
### exponential cost function
![exp_log](https://user-images.githubusercontent.com/98212546/224038634-fdebd9ee-38e3-4023-8ea6-bbfd54f6ee7d.png)
![exp_evol](https://user-images.githubusercontent.com/98212546/224038640-fc144ebe-ef2a-491c-912a-b055132d9d9c.png)

