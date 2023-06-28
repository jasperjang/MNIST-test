import PySimpleGUI as sg

from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate
from clenv.cli.queue.queue_manager import QueueManager
from git import Repo
from InquirerPy import prompt
from InquirerPy.validator import PathValidator, EmptyInputValidator
from collections import OrderedDict

sg.theme('DarkBlue15')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('CLENV', font='Ariel 24')],
            [sg.Text('Select a command', font='Ariel 18')],
            [sg.Button('config', font='Ariel 18'), sg.Button('task', font='Ariel 18'), sg.Button('user', font='Ariel 18')] ]

# Create the Window
mainWindow = sg.Window('CLENV', layout, modal=True)
# Event Loop to process "events" and get the "values" of the inputs
mainWindowActive = True
execWindowActive = False
execCompleteWindowActive = False
while True:
    mainEvent, mainValues = mainWindow.read()
    if mainEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
        break
    if mainWindowActive and mainEvent == 'task':
        execWindowActive = True

        queueManager = QueueManager()
        availableQueues = queueManager.get_available_queues()

        queueList = []
        for queue in availableQueues:
            queueList.append(f"{queue['name']} - idle workers: {[worker['name'] for worker in queue['workers'] if worker['task'] is None]} - total workers: {len(queue['workers'])}")
        
        taskTypes = ["training",
                     "testing",
                     "inference",
                     "data_processing",
                     "application",
                     "monitor",
                     "controller",
                     "optimizer",
                     "service",
                     "qc",
                     "other",]

        execLayout = [[sg.Text('Please choose a queue to execute the task')],
                      [sg.OptionMenu(queueList)],
                      [sg.Text('')],
                      [sg.Text('Please choose a task type')],
                      [sg.OptionMenu(taskTypes)],
                      [sg.Text('')],
                      [sg.Text('Please enter a task name')],
                      [sg.InputText('')],
                      [sg.Text('')],
                      [sg.Text('Please enter a script path')],
                      [sg.InputText('./')],
                      [sg.Button('Confirm'), sg.Button('Back')]]
        
        execWindow = sg.Window('Task Execution', execLayout, modal=True)
    if execWindowActive:
        execEvent, execValues = execWindow.read()
        if execEvent == 'Back' or execEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            execWindowActive = False
            execWindow.close()

        if execEvent == 'Confirm':
            queueName = ''
            for char in execValues[0]:
                if not char.isspace():
                    queueName += char
                else:
                    break
            queue = queueName
            taskType = execValues[1]
            taskName = execValues[2]
            path = execValues[3]
            
            repo = Repo(".")
            # Read the git information from current directory
            current_branch = repo.head.reference.name
            remote_url = repo.remotes.origin.url
            project_name = remote_url.split("/")[-1].split(".")[0]

            # Create a task object
            create_populate = CreateAndPopulate(
                project_name=project_name,
                task_name=taskName,
                task_type=taskType,
                repo=remote_url,
                branch=current_branch,
                # commit=args.commit,
                script=path,
                # working_directory=args.cwd,
                # packages=args.packages,
                # requirements_file=args.requirements,
                # docker=args.docker,
                # docker_args=args.docker_args,
                # docker_bash_setup_script=bash_setup_script,
                # output_uri=args.output_uri,
                # base_task_id=args.base_task_id,
                # add_task_init_call=not args.skip_task_init,
                # raise_on_missing_entries=True,
                verbose=True,
            )
            create_populate.create_task()

            create_populate.task._set_runtime_properties({"_CLEARML_TASK": True})

            task_id = create_populate.get_id()

            Task.enqueue(create_populate.task, queue_name=queue)

            execCompleteLayout = [[sg.Text(f"New task created id={task_id}")],
                                  [sg.Text(f"Task id={task_id} sent for execution on queue {queueName}")],
                                  [sg.Text("Execution log at:")],
                                  [sg.InputText(f'{create_populate.task.get_output_log_web_page()}')]]
        
            execCompleteWindow = sg.Window('Task Executing', execCompleteLayout, modal=True)
            execWindowActive = False
            execCompleteWindowActive = True
    if execCompleteWindowActive:
        execCompleteEvent, execCompleteValues = execCompleteWindow.read()
        if execCompleteEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            execCompleteWindowActive = False
            execWindowActive = True
            execCompleteWindow.close()

mainWindow.close()

