import PySimpleGUI as sg

from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate
from clenv.cli.queue.queue_manager import QueueManager
from git import Repo
from os.path import isfile
from config_manager import ConfigManager

################################################################################
######                         Helper Functions                            #####
################################################################################

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
    L = queueListItem.split(' ')
    queue = L[0]
    numIdleWorkers = len(L[4])
    totalWorkers = int(L[8])
    return queue, numIdleWorkers, totalWorkers

# checks if any values in the dictionary are empty
def checkBlankOptions(values):
    if (values['queueList'] == '' or
        values['taskTypes'] == '' or
        values['taskName'] == '' or
        len(values['path']) <= 2):
        return True
    return False

windowActivity = {'main':False,
                  'error':False,
                  'newConfig':False,
                  'actionSuccessful':False}

# sets active window to the inputted window name
def setActiveWindow(windowName):
    for window in windowActivity:
        if window == windowName:
            windowActivity[window] = True
        else:
            windowActivity[window] = False

# returns just the profile names from the list of non active profiles
def getNonActiveProfileNames(nonActiveProfiles):
    nonActiveProfileNames = []
    for profile in nonActiveProfiles:
        nonActiveProfileNames.append(profile['profile_name'])
    return nonActiveProfileNames

# returns list of profile names
def getProfileList(activeProfile, nonActiveProfiles):
    profileList = [activeProfile['profile_name']]
    for profile in nonActiveProfiles:
        profileList.append(profile["profile_name"])
    return profileList

# returns string of profiles from the list of profile names
def getProfileString(profileList):
    profileString = ''
    for profileIndex in range(len(profileList)):
        if profileIndex == 0:
            profileString += f'{profileList[0]} <active>\n'
        else:
            profileString += f'{profileList[profileIndex]}\n'
    return profileString

# creates an action success window
def actionSuccess():
    mainWindow['configOptions'].update('')
    actionSuccessfulLayout =    [
                                    [sg.Text('Action completed successfully!')]
                                ]
    actionSuccessfulWindow = sg.Window('', actionSuccessfulLayout, modal=True)
    setActiveWindow('actionSuccessful')
    return actionSuccessfulWindow

# gets direct
def getFileDir(path):
    dirPath = ''
    path = path.split('/')
    for i in range(len(path)-1):
        dirPath += f'{path[i]}/'
    return dirPath

################################################################################
######                          Initialization                             #####
################################################################################

sg.LOOK_AND_FEEL_TABLE['clearML'] = {'BACKGROUND': '#1A1E2C',
                                     'TEXT': '#ffffff',
                                     'INPUT': '#384161',
                                     'TEXT_INPUT': '#ffffff',
                                     'SCROLL': '#ffffff',
                                     'BUTTON': ('#ffffff', '#384161'),
                                     'PROGRESS': ('#ffffff', '#384161'),
                                     'BORDER': 1, 'SLIDER_DEPTH': 0, 
                                     'PROGRESS_DEPTH': 0}

sg.theme('clearML')

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
task_id = ''
queue = ''
URL = ''

configManager = ConfigManager('~/.clenv-config-index.json')
activeProfile = configManager.get_active_profile()[0]
nonActiveProfiles = configManager.get_non_active_profiles()
nonActiveProfileNames = getNonActiveProfileNames(nonActiveProfiles)
profileList = getProfileList(activeProfile, nonActiveProfiles)
profileString = getProfileString(profileList)

# Secondary layouts
mainLayout =    [
                    [sg.Button('Task Execution', font='Ariel 18', key='taskExec')],
                    [sg.Button('Profile Configuration', font='Ariel 18', key='config')]
                ]

execLayout =    [
                    [sg.Text('Please choose a queue to execute the task')],
                    [sg.OptionMenu(queueList, key='queueList')],
                    [sg.Text('')],
                    [sg.Text('Please choose a task type')],
                    [sg.OptionMenu(taskTypes, key='taskTypes')],
                    [sg.Text('')],
                    [sg.Text('Please enter a task name')],
                    [sg.InputText('', key='taskName')],
                    [sg.Text('')],
                    [sg.Text('Please enter a script path')],
                    [sg.InputText('./', key='path')],
                    [sg.Button('Confirm', key='execConfirm'), 
                     sg.Button('Back', key='execBack')]
                ]

execCompleteLayout =    [
                            [sg.Text(f"New task created id={task_id}", key='execCompleteT1')],
                            [sg.Text(f"Task id={task_id} sent for execution on queue {queue}", key='execCompleteT2')],
                            [sg.Text("Execution log at:")],
                            [sg.InputText(f'{URL}', key='execCompleteURL')],
                            [sg.Button('Back', key='execCompleteBack')]
                        ]

configLayout =  [
                    [sg.Text('Profile Configuration Options:')],
                    [sg.OptionMenu(['Checkout a Profile',
                                    'Create a Profile',
                                    'Delete a Profile',
                                    'List of Profiles',
                                    'Rename a Profile',
                                    'Configure API Path'], key='configOptions')],
                    [sg.Button('Confirm', key='configConfirm'), 
                     sg.Button('Back', key='configBack')]
                ]

configCheckoutLayout =  [
                            [sg.Text('Select a profile to checkout:')],
                            [sg.Text(f'Active Profile: {activeProfile}', key='checkoutAP')],
                            [sg.OptionMenu(nonActiveProfiles, key='checkoutNAP')],
                            [sg.Button('Confirm', key='configCheckoutConfirm'), 
                             sg.Button('Back', key='configCheckoutBack')]
                        ]

configCreateLayout =    [
                            [sg.Text('Enter a new profile name:')],
                            [sg.InputText('', key='newProfileName')],
                            [sg.Button('Confirm', key='configCreateConfirm'),
                             sg.Button('Back', key='configCreateBack')]
                        ]

configDeleteLayout =    [
                            [sg.Text('Select a profile to delete:')],
                            [sg.OptionMenu(nonActiveProfiles, key='deleteNAP')],
                            [sg.Button('Confirm', key='configDeleteConfirm'),
                             sg.Button('Back', key='configDeleteBack')]
                        ]

configListLayout =  [
                        [sg.Text('List of profiles:')],
                        [sg.Text(profileString, key='profileList')],
                        [sg.Button('Back', key='configListBack')]
                    ]

configRenameLayout =    [
                            [sg.Text('Select a profile to rename:')],
                            [sg.OptionMenu(profileList, key='profileListMenu')],
                            [sg.Text('')],
                            [sg.Text('Enter a new name:')],
                            [sg.InputText('', key='profileRename')],
                            [sg.Button('Confirm', key='configRenameConfirm'),
                             sg.Button('Back', key='configRenameBack')]
                        ]

configConfigureLayout =     [
                                [sg.Text('Select a profile to reconfigure:')],
                                [sg.OptionMenu(profileList, key='profileToConfig')],
                                [sg.Text('')],
                                [sg.Text('Enter a multiline configuration below:')],
                                [sg.Text('''
This can be found by navigating to the clearML website, 
clicking the button in the top right corner, then 
settings > workspace > create new credentials
                                         ''')],
                                [sg.Multiline('', key='multilineConfig', size=(60,9))],
                                [sg.Button('Confirm', key='configConfigureConfirm'),
                                 sg.Button('Back', key='configConfigureBack')]
                            ]

# Main layout
layout =    [
                [sg.Text('')],
                [sg.Image('./logo.png')],
                [sg.Text('')],
                [sg.Column(mainLayout, key='mainLayout'), 
                 sg.Column(execLayout, visible=False, key='execLayout'), 
                 sg.Column(execCompleteLayout, visible=False, key='execCompleteLayout'),
                 sg.Column(configLayout, visible=False, key='configLayout'),
                 sg.Column(configCheckoutLayout, visible=False, key='configCheckoutLayout'),
                 sg.Column(configCreateLayout, visible=False, key='configCreateLayout'),
                 sg.Column(configDeleteLayout, visible=False, key='configDeleteLayout'),
                 sg.Column(configListLayout, visible=False, key='configListLayout'),
                 sg.Column(configRenameLayout, visible=False, key='configRenameLayout'),
                 sg.Column(configConfigureLayout, visible=False, key='configConfigureLayout')]
            ]

mainWindow = sg.Window('CLENV', layout, modal=True, size=(600, 600), element_justification='c')
setActiveWindow('main')

################################################################################
######                             Main Loop                               #####
################################################################################

while True:
    if windowActivity['main']:
        mainEvent, mainValues = mainWindow.read()
        if mainEvent == sg.WIN_CLOSED: # if user closes window or clicks cancel
            break
        if mainEvent == 'taskExec':
            mainWindow['mainLayout'].update(visible=False)
            mainWindow['execLayout'].update(visible=True)
        if mainEvent == 'config':
            if not configManager.profile_has_initialized():
                newConfigLayout =   [
                                        [sg.Text('Please input a profile name:')],
                                        [sg.InputText()],
                                        [sg.Button('Confirm')]
                                    ]
                newConfigWindow = sg.Window('Profile Creation', newConfigLayout, modal=True)
                setActiveWindow('newConfig')
            else:
                mainWindow['mainLayout'].update(visible=False)
                mainWindow['configLayout'].update(visible=True)
        
        # config controllers
        if mainEvent == 'configConfirm':
            option = mainValues['configOptions']
            optLayout = f'config{option.split(" ")[0]}Layout'
            activeProfile = configManager.get_active_profile()[0]
            nonActiveProfiles = configManager.get_non_active_profiles()
            nonActiveProfileNames = getNonActiveProfileNames(nonActiveProfiles)
            profileList = getProfileList(activeProfile, nonActiveProfiles)
            if optLayout == 'configCheckoutLayout':
                mainWindow['checkoutAP'].update(f'Active Profile: {activeProfile["profile_name"]}')
                mainWindow['checkoutNAP'].update(values=nonActiveProfileNames)
            elif optLayout == 'configCreateLayout':
                mainWindow['newProfileName'].update('')
            elif optLayout == 'configDeleteLayout':
                mainWindow['deleteNAP'].update(values=nonActiveProfileNames)
                mainWindow['deleteNAP'].update('')
            elif optLayout == 'configListLayout':
                profileString = getProfileString(getProfileList(activeProfile, nonActiveProfiles))
                mainWindow['profileList'].update(profileString)
            elif optLayout == 'configRenameLayout':
                mainWindow['profileListMenu'].update(values=profileList)
                mainWindow['profileRename'].update('')
            elif optLayout == 'configConfigureLayout':
                mainWindow['profileToConfig'].update(values=profileList)
                mainWindow['profileToConfig'].update('')
                mainWindow['multilineConfig'].update('')
            mainWindow['configLayout'].update(visible=False)
            mainWindow[optLayout].update(visible=True)
        if mainEvent == 'configBack':
            mainWindow['configLayout'].update(visible=False)
            mainWindow['mainLayout'].update(visible=True)
        for optBack in ['configCheckoutBack',
                        'configCreateBack', 
                        'configDeleteBack', 
                        'configListBack', 
                        'configRenameBack', 
                        'configConfigureBack']:
            if mainEvent == optBack:
                mainWindow[f'{optBack.split("Back")[0]}Layout'].update(visible=False)
                mainWindow['configOptions'].update('')
                mainWindow['configLayout'].update(visible=True)
        if mainEvent == 'configCheckoutConfirm':
            profileName = mainValues['checkoutNAP']
            if configManager.has_profile(profile_name=profileName):
                configManager.set_active_profile(profileName)
                actionSuccessfulWindow = actionSuccess()
                mainWindow['configCheckoutLayout'].update(visible=False)
                mainWindow['configLayout'].update(visible=True)
        if mainEvent == 'configCreateConfirm':
            newProfileName = mainValues['newProfileName']
            if configManager.has_profile(profile_name=newProfileName):
                errorLayout =   [
                                    [sg.Text('Error: profile already exists', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                errorWindow = sg.Window('Error', errorLayout, modal=True)
                setActiveWindow('error')
            else:
                configManager.create_profile(newProfileName)
                actionSuccessfulWindow = actionSuccess()
                mainWindow['configCreateLayout'].update(visible=False)
                mainWindow['configLayout'].update(visible=True)
        if mainEvent == 'configDeleteConfirm':
            profileToDelete = mainValues['deleteNAP']
            configManager.delete_profile(profileToDelete)
            actionSuccessfulWindow = actionSuccess()
            mainWindow['configDeleteLayout'].update(visible=False)
            mainWindow['configLayout'].update(visible=True)
        if mainEvent == 'configRenameConfirm':
            profileToRename = mainValues['profileListMenu']
            profileRename = mainValues['profileRename']
            if profileRename in profileList:
                errorLayout =   [
                                    [sg.Text('Error: profile name is already taken', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                errorWindow = sg.Window('Error', errorLayout, modal=True)
                setActiveWindow('error')
            elif profileToRename == '' or profileRename == '':
                errorLayout =   [
                                    [sg.Text('Error: one or more options is blank', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                errorWindow = sg.Window('Error', errorLayout, modal=True)
                setActiveWindow('error')
            else:
                configManager.rename_profile(profileToRename, profileRename)
                actionSuccessfulWindow = actionSuccess()
                mainWindow['configRenameLayout'].update(visible=False)
                mainWindow['configLayout'].update(visible=True)
        if mainEvent == 'configConfigureConfirm':
            profileToConfig = mainValues['profileToConfig']
            config = mainValues['multilineConfig']
            configManager.reinitialize_api_config(profileToConfig, config)
            actionSuccessfulWindow = actionSuccess()
            mainWindow['configConfigureLayout'].update(visible=False)
            mainWindow['configLayout'].update(visible=True)

        # exec controllers
        if mainEvent == 'execBack':
            mainWindow['execLayout'].update(visible=False)
            mainWindow['mainLayout'].update(visible=True)
            mainWindow['queueList'].update('')
            mainWindow['taskTypes'].update('')
            mainWindow['taskName'].update('')
            mainWindow['path'].update('./')
        if mainEvent == 'execConfirm':
            rawQueueInfo = mainValues['queueList']
            taskType = mainValues['taskTypes']
            taskName = mainValues['taskName']
            path = mainValues['path']
            if checkBlankOptions(mainValues):
                errorLayout =   [
                                    [sg.Text('Error: one or more options is blank', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                errorWindow = sg.Window('Error', errorLayout, modal=True)
                setActiveWindow('error')
            elif not isfile(path):
                errorLayout =   [
                                    [sg.Text('Error: must input valid path', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                errorWindow = sg.Window('Error', errorLayout, modal=True)
                setActiveWindow('error')
            else:
                queue, numIdleWorkers, totalWorkers = getQueueInfo(rawQueueInfo)
                try:
                    repo = Repo(".")
                except:
                    errorLayout =   [
                                    [sg.Text('Error: no git repository detected \nat specified file directory', key='errorMsg')],
                                    [sg.Button('Back')]
                                ]
                    errorWindow = sg.Window('Error', errorLayout, modal=True)
                    setActiveWindow('error')
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
                URL = create_populate.task.get_output_log_web_page()
                mainWindow['execCompleteT1'].update(f"New task created id={task_id}")
                mainWindow['execCompleteT2'].update(f"Task id={task_id} sent for execution on queue {queue}")
                mainWindow['execCompleteURL'].update(f'{URL}')
                mainWindow['execLayout'].update(visible=False)
                mainWindow['execCompleteLayout'].update(visible=True)
        if mainEvent == 'execCompleteBack':
            mainWindow['queueList'].update('')
            mainWindow['taskTypes'].update('')
            mainWindow['taskName'].update('')
            mainWindow['path'].update('./')
            mainWindow['execCompleteLayout'].update(visible=False)
            mainWindow['execLayout'].update(visible=True)

    # error controllers
    if windowActivity['error']:
        errorEvent, errorValues = errorWindow.read()
        if errorEvent == 'Back' or errorEvent == sg.WIN_CLOSED:
            setActiveWindow('main')
            errorWindow.close()
    
    # newConfig controllers
    if windowActivity['newConfig']:
        newConfigEvent, newConfigValues = newConfigWindow.read()
        if newConfigEvent == 'Confirm':
            configManager.initialize_profile(f'{newConfigValues[0]}')
            activeProfiles = configManager.get_active_profile()
            nonActiveProfiles = configManager.get_non_active_profiles()
            mainWindow['mainLayout'].update(visible=False)
            mainWindow['configLayout'].update(visible=True)
            setActiveWindow('main')
            newConfigWindow.close()
        if newConfigEvent == sg.WIN_CLOSED:
            setActiveWindow('main')
            newConfigWindow.close()

    if windowActivity['actionSuccessful']:
        actionSuccessfulEvent, actionSuccessfulValues = actionSuccessfulWindow.read()
        if actionSuccessfulEvent == sg.WIN_CLOSED:
            setActiveWindow('main')
            actionSuccessfulWindow.close()

mainWindow.close()
