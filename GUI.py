import PySimpleGUI as sg

from clearml import Task
from clearml.backend_interface.task.populate import CreateAndPopulate
from clenv.cli.queue.queue_manager import QueueManager
from git import Repo
from os.path import isfile
from config_manager import ConfigManager
import webbrowser as wb
import os, json

################################################################################
######                         Helper Functions                           ######
################################################################################

# returns readable list of available queues
def get_queue_list():
    queue_manager = QueueManager()
    available_queues = queue_manager.get_available_queues()
    queue_list = []
    for queue in available_queues:
        queue_list.append(f"{queue['name']} - idle workers: {[worker['name'] for worker in queue['workers'] if worker['task'] is None]} - total workers: {len(queue['workers'])}")
    return queue_list

# returns queue name, number of idle workers, and total number of workers from the queue_list format above
def get_queue_info(queue_list_item):
    L = queue_list_item.split(' ')
    queue = L[0]
    num_idle_workers = len(L[4])
    total_workers = int(L[-1])
    return queue, num_idle_workers, total_workers

def get_queue_from_name(name, queue_list):
    for queue in queue_list:
        split_queue = queue.split(' ')
        if split_queue[0] == name:
            return queue

# checks if any values in the dictionary are empty
def check_blank_options(values):
    if (values['queue_list'] == '' or
        values['task_types'] == '' or
        values['task_name'] == '' or
        len(values['path']) <= 2):
        return True
    return False

window_activity = {'main':False,
                  'error':False,
                  'new_config':False,
                  'action_successful':False,
                  'new_template':False}

# sets active window to the inputted window name
def set_active_window(window_name):
    for window in window_activity:
        if window == window_name:
            window_activity[window] = True
        else:
            window_activity[window] = False

# returns just the profile names from the list of non active profiles
def get_non_active_profile_names(non_active_profiles):
    non_active_profile_names = []
    for profile in non_active_profiles:
        non_active_profile_names.append(profile['profile_name'])
    return non_active_profile_names

# returns list of profile names
def get_profile_list(active_profile, non_active_profiles):
    profile_list = [active_profile['profile_name']]
    for profile in non_active_profiles:
        profile_list.append(profile["profile_name"])
    return profile_list

# returns string of profiles from the list of profile names
def get_profile_string(profile_list):
    profile_string = ''
    for profile_index in range(len(profile_list)):
        if profile_index == 0:
            profile_string += f'{profile_list[0]} <active>\n'
        else:
            profile_string += f'{profile_list[profile_index]}\n'
    return profile_string

# creates an action success window
def action_success():
    main_window['config_options'].update('')
    action_successful_layout =    [
                                    [sg.Text('Action completed successfully!')]
                                ]
    action_successful_window = sg.Window('', action_successful_layout, modal=True)
    set_active_window('action_successful')
    return action_successful_window

# creates an error window
def create_error_window(message):
    error_layout =  [
        [sg.Text(f'Error: {message}', key='error_message')],
        [sg.Button('Back')]
                    ]
    error_window = sg.Window('Error', error_layout, modal=True)
    set_active_window('error')
    return error_window

# gets directory of given file path
def get_directory_path(path):
    dir_path = ''
    path_list = path.split('/')
    for item in path_list:
        if item == '':
            path_list.remove(item)
    for i in range(len(path_list)-1):
        dir_path += f'/{path_list[i]}'
    return dir_path

# returns script name from the path
def get_script_from_path(path):
    path_list = path.split('/')
    path_list.reverse()
    return path_list[0]

# returns a list of template names for the run templates menu
def get_template_names(current_templates):
    template_names = []
    for key in current_templates:
        template_names.append(key)
    return template_names

# executes the configuration from the run config
def exec_config(run_config):
    run_config['project_name'] = project_name,
    run_config['task_name'] = task_name,
    run_config['task_type'] = task_type,
    run_config['repo'] = remote_url,
    run_config['branch'] = current_branch,
    run_config['path'] = path,
    run_config['script'] = script,
    run_config['queue'] = queue
    create_populate = CreateAndPopulate(
        project_name=project_name,
        task_name=task_name,
        task_type=task_type,
        repo=remote_url,
        branch=current_branch,
        script=script,
        verbose=True,
    )
    create_populate.create_task()
    create_populate.task._set_runtime_properties({"_CLEARML_TASK": True})
    task_id = create_populate.get_id()
    Task.enqueue(create_populate.task, queue_name=queue)
    URL = create_populate.task.get_output_log_web_page()
    main_window['exec_complete_text1'].update(f"New task created id={task_id}")
    main_window['exec_complete_text2'].update(f"Task id={task_id} sent for execution on queue {queue}")
    main_window['exec_layout'].update(visible=False)
    main_window['exec_complete_layout'].update(visible=True)

################################################################################
######                          Initialization                            ######
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

queue_list = get_queue_list()
task_types = ["training",
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

config_manager = ConfigManager('~/.clenv-config-index.json')
active_profile = config_manager.get_active_profile()[0]
non_active_profiles = config_manager.get_non_active_profiles()
non_active_profile_names = get_non_active_profile_names(non_active_profiles)
profile_list = get_profile_list(active_profile, non_active_profiles)
profile_string = get_profile_string(profile_list)

# Secondary layouts
main_layout = [
    [sg.Button('    Task Execution    ', font='Ariel 18', key='task_exec')],
    [sg.Button('Profile Configuration', font='Ariel 18', key='config')]
]

run_template_layout = [
    [sg.Text('Select a template or run new:')],
    [sg.Listbox(['No templates created'], key='template_chosen', size=(60,9))],
    [sg.Button('Run New', key='run_template_new'),
     sg.Button('Run Template', key='run_template_template'),
     sg.Button('Delete Template', key='run_template_delete'),
     sg.Button('Back', key='run_template_back')]
]

exec_layout = [
    [sg.Text('Please choose a queue to execute the task')],
    [sg.OptionMenu(queue_list, key='queue_list')],
    [sg.Text('')],
    [sg.Text('Please choose a task type')],
    [sg.OptionMenu(task_types, key='task_types')],
    [sg.Text('')],
    [sg.Text('Please enter a task name')],
    [sg.InputText('', key='task_name')],
    [sg.Text('')],
    [sg.Text('Please enter a script path')],
    [sg.InputText('/', key='path')],
    [sg.Checkbox('Save as template?', key='save_as_template', visible=True)],
    [sg.Button('Confirm', key='exec_confirm'), 
     sg.Button('Back', key='exec_back')]
]

exec_complete_layout = [
    [sg.Text(f"New task created id={task_id}", key='exec_complete_text1')],
    [sg.Text(f"Task id={task_id} sent for execution on queue {queue}", key='exec_complete_text2')],
    [sg.Text("Execution log at:")],
    [sg.Button('Navigate to project on clearML', key='exec_complete_URL')],
    [sg.Button('Back', key='exec_complete_back')]
]

config_layout = [
    [sg.Text('Profile Configuration Options:')],
    [sg.OptionMenu(['Checkout a Profile',
                    'Create a Profile',
                    'Delete a Profile',
                    'List of Profiles',
                    'Rename a Profile',
                    'Configure API Path'], key='config_options')],
    [sg.Button('Confirm', key='config_confirm'), 
     sg.Button('Back', key='config_back')]
]

config_checkout_layout = [
    [sg.Text('Select a profile to checkout:')],
    [sg.Text(f'Active Profile: {active_profile}', key='checkout_active_profile')],
    [sg.OptionMenu(non_active_profiles, key='checkout_non_active_profiles')],
    [sg.Button('Confirm', key='config_checkout_confirm'), 
     sg.Button('Back', key='config_checkout_back')]
]

config_create_layout = [
    [sg.Text('Enter a new profile name:')],
    [sg.InputText('', key='new_profile_name')],
    [sg.Button('Confirm', key='config_create_confirm'),
     sg.Button('Back', key='config_create_back')]
]

config_delete_layout = [
    [sg.Text('Select a profile to delete:')],
    [sg.OptionMenu(non_active_profiles, key='delete_non_active_profiles')],
    [sg.Button('Confirm', key='config_delete_confirm'),
     sg.Button('Back', key='config_delete_back')]
]

config_list_layout = [
    [sg.Text('List of profiles:')],
    [sg.Text(profile_string, key='profile_list')],
    [sg.Button('Back', key='config_list_back')]
]

config_rename_layout = [
    [sg.Text('Select a profile to rename:')],
    [sg.OptionMenu(profile_list, key='profile_list_menu')],
    [sg.Text('')],
    [sg.Text('Enter a new name:')],
    [sg.InputText('', key='profile_rename')],
    [sg.Button('Confirm', key='config_rename_confirm'),
     sg.Button('Back', key='config_rename_back')]
]

config_configure_layout = [
    [sg.Text('Select a profile to reconfigure:')],
    [sg.OptionMenu(profile_list, key='profile_to_config')],
    [sg.Text('')],
    [sg.Text('Enter a multiline configuration below:')],
    [sg.Text('''
This can be found by navigating to the clearML website, 
clicking the button in the top right corner, then 
settings > workspace > create new credentials
             ''')],
    [sg.Multiline('', key='multiline_config', size=(60,9))],
    [sg.Button('Confirm', key='config_configure_confirm'),
     sg.Button('Back', key='config_configure_back')]
]

# Main layout
layout = [
    [sg.Text('')],
    [sg.Image('./logo.png')],
    [sg.Text('')],
    [sg.Column(main_layout, key='main_layout'), 
     sg.Column(exec_layout, visible=False, key='exec_layout'), 
     sg.Column(exec_complete_layout, visible=False, key='exec_complete_layout'),
     sg.Column(config_layout, visible=False, key='config_layout'),
     sg.Column(config_checkout_layout, visible=False, key='config_checkout_layout'),
     sg.Column(config_create_layout, visible=False, key='config_create_layout'),
     sg.Column(config_delete_layout, visible=False, key='config_delete_layout'),
     sg.Column(config_list_layout, visible=False, key='config_list_layout'),
     sg.Column(config_rename_layout, visible=False, key='config_rename_layout'),
     sg.Column(config_configure_layout, visible=False, key='config_configure_layout'),
     sg.Column(run_template_layout, visible=False, key='run_template_layout')]
]

main_window = sg.Window('CLENV', layout, modal=True, size=(600, 600), element_justification='c')
set_active_window('main')

################################################################################
######                             Main Loop                              ######
################################################################################

while True:
    if window_activity['main']:
        main_event, main_values = main_window.read()
        if main_event == sg.WIN_CLOSED: # if user closes window or clicks cancel
            break
        if main_event == 'task_exec':
            if not isfile('./task_templates.json'):
                with open('task_templates.json', 'w') as f:
                    json.dump({}, f, indent=4)
            with open("task_templates.json", "r") as f:
                current_templates = json.load(f)
            template_names = get_template_names(current_templates)
            main_window['template_chosen'].update(values=template_names)
            main_window['main_layout'].update(visible=False)
            main_window['run_template_layout'].update(visible=True)
        if main_event == 'config':
            if not config_manager.profile_has_initialized():
                new_config_layout = [
                    [sg.Text('Please input a profile name:')],
                    [sg.InputText()],
                    [sg.Button('Confirm')]
                ]
                new_config_window = sg.Window('Profile Creation', new_config_layout, modal=True)
                set_active_window('new_config')
            else:
                main_window['main_layout'].update(visible=False)
                main_window['config_layout'].update(visible=True)
        
        # config controllers
        if main_event == 'config_confirm':
            option = main_values['config_options']

            # option_layout is the string key associated with the layout of the option selected in the dropdown menu
            option_layout = f'config_{option.split(" ")[0].lower()}_layout'

            active_profile = config_manager.get_active_profile()[0]
            non_active_profiles = config_manager.get_non_active_profiles()
            non_active_profile_names = get_non_active_profile_names(non_active_profiles)
            profile_list = get_profile_list(active_profile, non_active_profiles)

            # reset layout whenever selected
            if option_layout == 'config_checkout_layout':
                main_window['checkout_active_profile'].update(f'Active Profile: {active_profile["profile_name"]}')
                main_window['checkout_non_active_profiles'].update(values=non_active_profile_names)
            elif option_layout == 'config_create_layout':
                main_window['new_profile_name'].update('')
            elif option_layout == 'config_delete_layout':
                main_window['delete_non_active_profiles'].update(values=non_active_profile_names)
                main_window['delete_non_active_profiles'].update('')
            elif option_layout == 'config_list_layout':
                profile_string = get_profile_string(get_profile_list(active_profile, non_active_profiles))
                main_window['profile_list'].update(profile_string)
            elif option_layout == 'config_rename_layout':
                main_window['profile_list_menu'].update(values=profile_list)
                main_window['profile_rename'].update('')
            elif option_layout == 'config_configure_layout':
                main_window['profile_to_config'].update(values=profile_list)
                main_window['profile_to_config'].update('')
                main_window['multiline_config'].update('')
            
            main_window['config_layout'].update(visible=False)
            main_window[option_layout].update(visible=True)
        if main_event == 'config_back':
            main_window['config_layout'].update(visible=False)
            main_window['main_layout'].update(visible=True)
        for option_back in ['config_checkout_back',
                            'config_create_back', 
                            'config_delete_back', 
                            'config_list_back', 
                            'config_rename_back', 
                            'config_configure_back']:
            if main_event == option_back:
                main_window[f'{option_back.split("back")[0]}layout'].update(visible=False)
                main_window['config_options'].update('')
                main_window['config_layout'].update(visible=True)
        if main_event == 'config_checkout_confirm':
            profileName = main_values['checkout_non_active_profiles']
            if config_manager.has_profile(profile_name=profileName):
                config_manager.set_active_profile(profileName)
                action_successful_window = action_success()
                main_window['config_checkout_layout'].update(visible=False)
                main_window['config_layout'].update(visible=True)
        if main_event == 'config_create_confirm':
            new_profile_name = main_values['new_profile_name']
            if config_manager.has_profile(profile_name=new_profile_name):
                error_window = create_error_window('profile already exists')
            else:
                config_manager.create_profile(new_profile_name)
                action_successful_window = action_success()
                main_window['config_create_layout'].update(visible=False)
                main_window['config_layout'].update(visible=True)
        if main_event == 'config_delete_confirm':
            profile_to_delete = main_values['delete_non_active_profiles']
            config_manager.delete_profile(profile_to_delete)
            action_successful_window = action_success()
            main_window['config_delete_layout'].update(visible=False)
            main_window['config_layout'].update(visible=True)
        if main_event == 'config_rename_confirm':
            profile_to_rename = main_values['profile_list_menu']
            profile_rename = main_values['profile_rename']
            if profile_rename in profile_list:
                error_window = create_error_window('profile name is already taken')
            elif profile_to_rename == '' or profile_rename == '':
                error_window = create_error_window('one or more options is blank')
            else:
                config_manager.rename_profile(profile_to_rename, profile_rename)
                action_successful_window = action_success()
                main_window['config_rename_layout'].update(visible=False)
                main_window['config_layout'].update(visible=True)
        if main_event == 'config_configure_confirm':
            profile_to_config = main_values['profile_to_config']
            config = main_values['multiline_config']
            try:
                config_manager.reinitialize_api_config(profile_to_config, config)
                action_successful_window = action_success()
                main_window['config_configure_layout'].update(visible=False)
                main_window['config_layout'].update(visible=True)
            except:
                error_window = create_error_window('invalid configuration format')

        # run template controllers
        if main_event == 'run_template_new':
            main_window['save_as_template'].update(False)
            main_window['save_as_template'].update(visible=True)
            main_window['run_template_layout'].update(visible=False)
            main_window['exec_layout'].update(visible=True)
        if main_event == 'run_template_template':
            if main_values['template_chosen'] != {} and main_values['template_chosen'] != []:
                template_name = main_values['template_chosen'][0]
                with open("task_templates.json", "r") as f:
                    current_templates = json.load(f)
                template = current_templates[template_name]
                queue_list = get_queue_list()
                queue = get_queue_from_name(template['queue'], queue_list)
                main_window['queue_list'].update(queue)
                main_window['task_types'].update(f'{template["task_type"]}')
                main_window['task_name'].update(f'{template["task_name"]}')
                main_window['path'].update(f'{template["path"]}')
                main_window['save_as_template'].update(False)
                main_window['save_as_template'].update(visible=False)
                main_window['run_template_layout'].update(visible=False)
                main_window['exec_layout'].update(visible=True)
            else:
                error_window = create_error_window('no template selected')
        if main_event == 'run_template_delete':
            if main_values['template_chosen'] != {} and main_values['template_chosen'] != []:
                template_name = main_values['template_chosen'][0]
                with open("task_templates.json", "r") as f:
                    current_templates = json.load(f)
                current_templates.pop(template_name)
                with open("task_templates.json", "w") as f:
                    json.dump(current_templates, f)
                template_names = get_template_names(current_templates)
                main_window['template_chosen'].update(values=template_names)
            else:
                error_window = create_error_window('no template selected')
        if main_event == 'run_template_back':
            main_window['run_template_layout'].update(visible=False)
            main_window['main_layout'].update(visible=True)

        # exec controllers
        if main_event == 'exec_back':
            main_window['exec_layout'].update(visible=False)
            main_window['run_template_layout'].update(visible=True)
            main_window['queue_list'].update('')
            main_window['task_types'].update('')
            main_window['task_name'].update('')
            main_window['path'].update('/')
        if main_event == 'exec_confirm':
            raw_queue_info = main_values['queue_list']
            task_type = main_values['task_types']
            task_name = main_values['task_name']
            path = main_values['path']
            if check_blank_options(main_values):
                error_window = create_error_window('one or more options is blank')
            elif not isfile(path):
                error_window = create_error_window('must input valid path')
            else:
                queue, num_idle_workers, total_workers = get_queue_info(raw_queue_info)
                try:
                    dir_path = get_directory_path(path)
                    repo = Repo(f'{dir_path}')
                except:
                    error_window = create_error_window('no git repository detected \nat specified file directory')
                # Read the git information from current directory
                current_branch = repo.head.reference.name
                remote_url = repo.remotes.origin.url
                project_name = remote_url.split("/")[-1].split(".")[0]
                script = get_script_from_path(path)
                run_config = {
                        'project_name':project_name,
                        'task_name':task_name,
                        'task_type':task_type,
                        'repo':remote_url,
                        'branch':current_branch,
                        'path':path,
                        'script':script,
                        'queue':queue
                    }
                if main_values['save_as_template']:  
                    new_template_layout = [
                        [sg.Text('Please input a template name:')],
                        [sg.InputText()],
                        [sg.Button('Confirm')]
                    ]
                    new_template_window = sg.Window('Template Creation', new_template_layout, modal=True)
                    set_active_window('new_template')
                else:
                    exec_config(run_config)
        if main_event == 'exec_complete_URL':
            wb.open(URL)
        if main_event == 'exec_complete_back':
            main_window['queue_list'].update('')
            main_window['task_types'].update('')
            main_window['task_name'].update('')
            main_window['path'].update('/')
            main_window['save_as_template'].update(False)
            main_window['exec_complete_layout'].update(visible=False)
            main_window['run_template_layout'].update(visible=True)

    # error controllers
    if window_activity['error']:
        error_event, error_values = error_window.read()
        if error_event == 'Back' or error_event == sg.WIN_CLOSED:
            set_active_window('main')
            error_window.close()
    
    # new_config controllers
    if window_activity['new_config']:
        new_config_event, new_config_values = new_config_window.read()
        if new_config_event == 'Confirm':
            config_manager.initialize_profile(f'{new_config_values[0]}')
            active_profiles = config_manager.get_active_profile()
            non_active_profiles = config_manager.get_non_active_profiles()
            main_window['main_layout'].update(visible=False)
            main_window['config_layout'].update(visible=True)
            set_active_window('main')
            new_config_window.close()
        if new_config_event == sg.WIN_CLOSED:
            set_active_window('main')
            new_config_window.close()

    # new template controllers
    if window_activity['new_template']:
        new_template_event, new_template_values = new_template_window.read()
        if new_template_event == 'Confirm':
            template_name = new_template_values[0]
            with open("task_templates.json", "r") as f:
                current_templates = json.load(f)
            current_templates[template_name] = run_config
            with open("task_templates.json", "w") as f:
                json.dump(current_templates, f, indent=4)
            template_names = get_template_names(current_templates)
            main_window['template_chosen'].update(values=template_names)
            set_active_window('main')
            new_template_window.close()
            exec_config(run_config)
        if new_template_event == sg.WIN_CLOSED:
            set_active_window('main')
            new_template_window.close()

    if window_activity['action_successful']:
        action_successful_event, action_successful_values = action_successful_window.read()
        if action_successful_event == sg.WIN_CLOSED:
            set_active_window('main')
            action_successful_window.close()

main_window.close()
