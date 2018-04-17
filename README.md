

# Egocentric and allocentric frame references in rodent spatial navitation 

## Basic idea
The goal of this project is to investigate how spatial navigation works in rodents. Our approach is functionalist 
in the sense that we believe that by building a system that performs the task, we might be in a better place to 
understand what the mechanisms used in biology might look like or what are the restrictions that might be 
involved. Therefore, we will model rodent spatial navigation at a high level of abstraction but keeping a certain 
level of resemblence to what is known about the navigation system in rodents.

## Assumptions and restrictions

**Egocentric information** proprioceptive information (angular velocity, motion speed, etc). It is used to perform
path integration.


**Allocentric information** GPS-like information (location in the environment). It is used as a map.

_Add references and a literature review of models of spatial navigation_

## Methods
We have an agent that has an _egocentric_ and an _allocentric_ reference frames that has to navigate in an 
environment. The egocentric reference frame that provides information in reference to its own body 
(e.g. wall to my right) while the allocentric reference frame provides spatial information with respect to the 
environment (e.g. headed in the north direction).

The enviroment consists in a grid-like arena with walls, lava and rewards. Each state that the agent can be in
is a spatial location in the arena (cell). The navigation task of the agent consists to explore the arena to look for 
reward cells while avoiding bumping into walls and stepping into lava cells. 

In order to study how the egocentric and allocentric interact, we allow these two systems to exchange 
information. On the one hand, 


## Questions 

- how the two refernece frames interact: 
    - are they both informing each other constantly?
    - can they both predict each other states?
    
- When and how any of the reference frames is preferred?
 
- can egocentric information transformed into allocentric information (and viceversa)?

### TO DO

- Bibliography review
    - models of rodent spatial navigation
- Design how to implement the egocentric/allocentric reference frames
- 
    

## Code


![Class diagram](https://github.com/edgarbc/janus/blob/commented-version/img/class_diagram.svg)


- Agent class

- Policy class

- Task class

### Usage

```
python demo.py
```    
