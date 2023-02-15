# Human-Agent Teamwork for Search and Rescue
This is the repository for the interdependence and trust assignments of the course on Collaborative AI at the TU Delft. The repository uses the [MATRX software package](https://matrx-software.com/) to create a simulated search and rescue task in a two-dimensional grid environment. The environment consists of multiple areas, injured victims, and obstacles blocking area entrances. One artificial agent (called RescueBot) and one human agent need to rescue these victims and deliver them to a drop off zone, while communicating and collaborating with each other.

![environment-chat-1](https://user-images.githubusercontent.com/54837051/204800699-89ed7159-d329-4f95-8441-acb601ff90a5.png)
## Task
The objective of the task is to find eight target victims in the different areas and carry them to the drop zone. Rescuing mildly injured victims (yellow color) adds three points to the total score, rescuing critically injured victims (red color) adds six points to the total score. The world terminates after successfully rescuing all target victims, the corresponding output logs will then be saved in the 'logs' folder. We created three human capability conditions: strong, weak, and normal. These capabilities result in different interdependence relationships between RescueBot and the human agent. Below we list the common and unique capabilities for each of these conditions.
#### Common capabilities across all conditions
- Critically injured victims can only be carried by both human and RescueBot together. 
- RescueBot can carry mildly injured victims alone, but doing this together with human assistance is much faster.
- The big grey rock can only be removed by both human and RescueBot together.
- RescueBot can remove the small brown stone alone, but doing this together with human assistance is much faster.
- The tree can only be removed by RescueBot.
- RescueBot can only carry one victim at the same time.
#### Capabilities for the strong human
- The strong human can identify obstacles with a higher perception range of 10 grid cells.
- The strong human can carry all mildly injured victims at the same time.
- The strong human can remove the small brown stone alone, but doing this together with RescueBot is much faster.
#### Capabilities for the weak human
- The weak human is unable to remove the small brown stone alone and needs assistance from RescueBot to remove this obstacle together.
- The weak human can identify obstacles with a normal perception range of 1 grid cell.
- The weak human is unable to carry mildly injured victims alone and needs assistance from RescueBot to carry this victim together.
- The weak human can only carry one victim at the same time.
#### Capabilities from the normal human
- The normal human can identify obstacles with a normal perception range of 1 grid cell.
- The normal human can remove the small brown stone alone, but doing this together with RescueBot is much faster.
- The normal human can rescue mildly injured victims alone.
- The normal human can carry only one victim at the same time.
## Installation
Download or clone this repository and the required dependencies listed in the 'requirements.txt' file. We recommend the use of Python 3.8 or 3.9, and to create a virtual environment for this project. You can use the following step by step installation steps after cloning or downloading this repository:
- Install the required dependencies through 'pip install -r requirements.txt'. 
- Launch the human-agent teamwork task by running main.py.
- You will be asked to enter which task type to run: 
  - 'tutorial' will launch a step by step tutorial of the task in a simplified and smaller world, aimed at getting you familiar with the environment, controls, and messaging system. We highly recommend you to start with this tutorial.
  - 'official' will launch the complete task. Next, you will be asked to enter a name or id for the human agent that you will control. Finally, you will be asked to enter one of the human capability conditions 'normal', 'strong', or 'weak'. 
- Go to http://localhost:3000 and clear your old cache of the page by pressing 'ctrl' + 'F5'.
- Open the 'God' and human agent view. Start the task in the 'God' view with the play icon in the top right of the toolbar. The 'God' view is shown in the image above, cannot be used to control agents, and should only be used for debugging purposes. 
- Go to the human agent view to start the task. Open the messaging interface by pressing the chat box icon in the top right of the toolbar. You can now start playing the task.
## Overview
Below we discuss the content and files of the important folders in more detail. For the assignments, the only implemantation modifactions should be made to the 'agents1' and optionally the 'brains1' and 'worlds1' folders.
- 'actions1': Contains the 'CustomActions.py' file defining the various customized actions like 'CarryObjectTogether' and 'DropObjectTogether'.
- 'agents1': Contains the 'OfficialAgent.py' and 'TutorialAgent.py' files defining the behavior of the agents for the official and tutorial tasks. For the trust assigment, you will extend and modify the 'OfficialAgent.py'. More specifcally, you will extend the function '_trustBelief' and use the outputs of this function to adapt the agent's behavior defined by the function 'decide_on_actions'. 
- 'beliefs': Contains the 'currentTrustBelief.csv' and 'allTrustBeliefs.csv' files. These files are used for retrieving trust belief values when interacting with a human more than once, and used to save trust belief values for all the human agents that RescueBot collaborated with. 
- 'brains1': Contains the 'ArtificialBrain.py' and 'HumanBrain.py' files required to initialize RescueBot and the human agent. For the trust assignment, you might modify the human brain to create slower or faster humans, for example.
- 'loggers': Contains the 'ActionLogger.py' and 'OutputLogger.py' files. The action logger saves the actions and locations of both human and RescueBot during every tick of the task. In the MATRX world, all time is measured in ticks instead of seconds, and actions and messages are all executed at a single tick. The tick duration is set at 0.1, which means around 10 ticks are executed in a second. In addition, the output logger creates one output file and line with the time it took to finish the task (in ticks) and the total number of human and agent actions during the task. Finally, the output logger saves the trust belief values to the 'allTrustBeliefs.csv' file mentioned above. It is important to know that the output logger is only called when the task is successfully completed, or when you press the stop button in the 'God' view (the square button next to the play button). 
- 'worlds1': Contains the 'WorldBuilder.py' file defining the search and rescue environment and task. For the trust assignment, you might modify the world builder to add slower or faster humans, for example. 

## More information
[More documentation can be found here](https://tracinsy.ewi.tudelft.nl/pubtrac/BW4T-Matrx-CollaborativeAI/wiki). This page contains documentation information related to the assignment from last years, so not all information is relevant. However, we believe some information can still be relevant. Finally, [MATRX documentation information can be found here](http://docs.matrx-software.com/en/master/), [MATRX tutorials can be found here](https://matrx-software.com/tutorials/), and the [MATRX GitHub page here](https://github.com/matrx-software/matrx).
