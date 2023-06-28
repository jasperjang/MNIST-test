import PySimpleGUI as sg

from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate
from clenv.cli.queue.queue_manager import QueueManager
from git import Repo
from os.path import isfile

# returns readable list of available queues
def getQueueList():
    queueManager = QueueManager()
    availableQueues = queueManager.get_available_queues()
    queueList = []
    for queue in availableQueues:
        queueList.append(f"{queue['name']} - idle workers: {[worker['name'] for worker in queue['workers'] if worker['task'] is None]} - total workers: {len(queue['workers'])}")
    return queueList

# returns queue name, number of idle workers, and total number of workers from the queueList format above
def getQueueInfo(queueListItem):
    L = queueListItem.split()
    queue = L[0]
    numIdleWorkers = len(L[4])
    totalWorkers = int(L[8])
    return queue, numIdleWorkers, totalWorkers

windowActivity = {'main':False,
                  'exec':False,
                  'execComplete':False,
                  'pathError':False}

# sets active window to the inputted window name
def setActiveWindow(windowName):
    for window in windowActivity:
        if window == windowName:
            windowActivity[window] = True
        else:
            windowActivity[window] = False

# checks if any values in the dictionary are empty
def checkBlankOptions(dict):
    for key in dict:
        value = dict[key]
        if value == '' or value == [] or value == {}:
            return True
        else:
            continue
    return False

sg.theme('DarkBlue15')   # Add a touch of color
# All the stuff inside your window.
layout = [  [sg.Text('CLENV', font='Ariel 18')],
            [sg.Text('Select a command', font='Ariel 14')],
            [sg.Button('config', font='Ariel 14'), sg.Button('task', font='Ariel 14'), sg.Button('user', font='Ariel 14')] ]

# Create the Window
mainWindow = sg.Window('CLENV', layout, modal=True)
# Event Loop to process "events" and get the "values" of the inputs
setActiveWindow('main')
while True:
    if windowActivity['main']:
        mainEvent, mainValues = mainWindow.read()
        if mainEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            break
        if mainEvent == 'task':
            setActiveWindow('exec')
        
            queueList = getQueueList()
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

    if windowActivity['exec']:
        execEvent, execValues = execWindow.read()
        if execEvent == 'Back' or execEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            setActiveWindow('main')
            execWindow.close()

        if execEvent == 'Confirm':
            rawQueueInfo = execValues[0]
            taskType = execValues[1]
            taskName = execValues[2]
            path = execValues[3]
            if checkBlankOptions(execValues):
                pathErrorLayout = [[sg.Text('Error: must input valid path')],
                                   [sg.Button('Back')]]
                pathErrorWindow = sg.Window('Error', pathErrorLayout, modal=True)
                setActiveWindow('pathError')
            elif not isfile(path):
                pathErrorLayout = [[sg.Text('Error: must input valid path')],
                                   [sg.Button('Back')]]
                pathErrorWindow = sg.Window('Error', pathErrorLayout, modal=True)
                setActiveWindow('pathError')
            else:
                queue, numIdleWorkers, totalWorkers = getQueueInfo(rawQueueInfo)
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
                                      [sg.Text(f"Task id={task_id} sent for execution on queue {queue}")],
                                      [sg.Text("Execution log at:")],
                                      [sg.InputText(f'{create_populate.task.get_output_log_web_page()}')]]
            
                execCompleteWindow = sg.Window('Task Executing', execCompleteLayout, modal=True)
                setActiveWindow('execComplete')
    if windowActivity['execComplete']:
        execCompleteEvent, execCompleteValues = execCompleteWindow.read()
        if execCompleteEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            setActiveWindow('exec')
            execCompleteWindow.close()
    if windowActivity['pathError']:
        pathErrorEvent, pathErrorValues = pathErrorWindow.read()
        if pathErrorEvent == 'Back' or pathErrorEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            setActiveWindow('exec')
            pathErrorWindow.close()

mainWindow.close()

