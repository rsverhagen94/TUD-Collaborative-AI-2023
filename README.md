# Human-Agent Teamwork for Search and Rescue
## Installation
- Install the required dependencies through 'pip install -r requirements.txt'. 
- Launch the human-agent teamwork task by running main.py.
- You will be asked to enter which task type to run: 
  - 'tutorial' will launch a step by step tutorial of the task in a simplified and smaller world, aimed at getting you familiar with the environment, controls, and messaging system. 
  - 'official' will launch the complete task. First, you will be asked to enter a name or id for the human agent that you will control. Next, you will be asked to enter one of the human capability conditions 'normal', 'strong' (better vision and carrying capabilities), or 'weak' (worse removing and carrying capabilities). 
- Go to http://localhost:3000 and clear your old cache of the page by pressing 'ctrl' + 'F5'.
- Open the 'God' and human agent view. Start the task in the 'God' view with the play icon in the top right of the toolbar. 
- Go to the human agent view to start the task. Open the messaging interface by pressing the chat box icon in the top right of the toolbar. 

## Task
The objective of the task is to find target victims in the different areas and carry them to the drop zone. Rescuing mildly injured victims (yellow color) adds three points to the total score, rescuing critically injured victims (red color) adds six points to the total score. Critically injured victims can only be carried by both human and agent together. Areas can be blocked by three different obstacle types. One of these can only be removed together, one only by the agent, and one both alone and together (but together is much faster). The world terminates after successfully rescuing all target victims. Save the output logs by pressing the stop icon in the 'God' view, which can then be found in the 'logs' folder. The image below shows the 'God' view and the messaging interface. 

![environment-chat-1](https://user-images.githubusercontent.com/54837051/204800699-89ed7159-d329-4f95-8441-acb601ff90a5.png)
